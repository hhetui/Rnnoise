/* Copyright (c) 2018 Gregor Richards
 * Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "rnnoise.h"
#include "pitch.h"
#include "arch.h"
#include "rnn.h"
#include "rnn_data.h"

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (20<<FRAME_SIZE_SHIFT) //480
#define WINDOW_SIZE (2*FRAME_SIZE)  //960
#define FREQ_SIZE (FRAME_SIZE + 1)  //481

#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE) //1728

#define SQUARE(x) ((x)*(x))

#define NB_BANDS 14

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)


#ifndef TRAINING
#define TRAINING 0//我自己改为了1 方便看代码  本来是0
#endif


/* The built-in model, used if no file is given as input */
extern const struct RNNModel rnnoise_model_orig;


static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};

/*
CommonState含有一个kfft  一个长度为480的数组 一个22*22 的二维数组 都是装的sin cos之类的数值
*/
typedef struct {
  int init;
  kiss_fft_state *kfft;
  float half_window[FRAME_SIZE]; //480
  float dct_table[NB_BANDS*NB_BANDS]; //22*22
} CommonState;

//保存了一堆数组 有关音频的各种参数 
struct DenoiseState {
  float analysis_mem[FRAME_SIZE]; //480
  float cepstral_mem[CEPS_MEM][NB_BANDS];//8 x 22
  int memid;
  float synthesis_mem[FRAME_SIZE]; //480
  float pitch_buf[PITCH_BUF_SIZE]; //1728=768+960
  float pitch_enh_buf[PITCH_BUF_SIZE];//1728=768+960
  float last_gain;
  int last_period;
  float mem_hp_x[2]; // 2
  float lastg[NB_BANDS]; //22
  RNNState rnn;
};

//计算各个频段的能量(X的各个元素的平方和 加权滚动求和)存入bandE
void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
      tmp += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

//计算各个频段的能量（X和P两个的点乘，滚动求和）存入bandE
void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r;
      tmp += X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i;
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

//用计算的能量bandE计算各个收益 g
void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
    }
  }
}


CommonState common;

//将common 初始化为论文中的 sin cos系数数组  half_window和dct_table
static void check_init() {
  int i;
  if (common.init) return;
  common.kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE, NULL, NULL, NULL, 0);
  for (i=0;i<FRAME_SIZE;i++)
    common.half_window[i] = sin(.5*M_PI*sin(.5*M_PI*(i+.5)/FRAME_SIZE) * sin(.5*M_PI*(i+.5)/FRAME_SIZE));
  for (i=0;i<NB_BANDS;i++) {
    int j;
    for (j=0;j<NB_BANDS;j++) {
      common.dct_table[i*NB_BANDS + j] = cos((i+.5)*j*M_PI/NB_BANDS);
      if (j==0) common.dct_table[i*NB_BANDS + j] *= sqrt(.5);
    }
  }
  common.init = 1;
}

//将in 与common的dct_table 的每22个对位相乘 求和*sqrt(2./22) 存为out[i] out长22
static void dct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[j*NB_BANDS + i];
    }
    out[i] = sum*sqrt(2./22);
  }
}


//新建kiss_fft_cpx x和y数组 将in存入x的r 然后对x和y用common.kfft计算opu_fft
//输出长度为481的out kiss_fft_cpx 二维数组
static void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  opus_fft(common.kfft, x, y, 0);
  //为什么得到481长度的out 貌似是对称取一半即可
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

//逆变换
static void inverse_transform(float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  opus_fft(common.kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}

//对x数组与common.half_window对位相乘
//x长度960 对称操作
static void apply_window(float *x) {
  int i;
  check_init();
  for (i=0;i<FRAME_SIZE;i++) {
    x[i] *= common.half_window[i];
    x[WINDOW_SIZE - 1 - i] *= common.half_window[i];
  }
}

//DenoiseState的大小
int rnnoise_get_size() {
  return sizeof(DenoiseState);
}

//初始化st 如果给的model非空就st.rnn.model=model 否则=rnnoise_model_orig
//并且为st.rnn申请三个gru的内存
int rnnoise_init(DenoiseState *st, RNNModel *model) {
  memset(st, 0, sizeof(*st));
  if (model)
    st->rnn.model = model;
  else
    st->rnn.model = &rnnoise_model_orig;
  st->rnn.vad_gru_state = calloc(sizeof(float), st->rnn.model->vad_gru_size);
  st->rnn.noise_gru_state = calloc(sizeof(float), st->rnn.model->noise_gru_size);
  st->rnn.denoise_gru_state = calloc(sizeof(float), st->rnn.model->denoise_gru_size);
  return 0;
}

//新建一个DenoiseState st并且将其rnn初始化为model 然后返回st
DenoiseState *rnnoise_create(RNNModel *model) {
  DenoiseState *st;
  st = malloc(rnnoise_get_size());
  rnnoise_init(st, model);
  return st;
}

//释放DenoiseState st 内存
void rnnoise_destroy(DenoiseState *st) {
  free(st->rnn.vad_gru_state);
  free(st->rnn.noise_gru_state);
  free(st->rnn.denoise_gru_state);
  free(st);
}

#if TRAINING
int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;
#endif

//新建长度为in两倍的数组x  把st.analysis_mem 和 in 拼接 为 x
//把in覆盖 st的 analysis_mem（即每次其包含上次的in的信息）
//对拼好的x进行apply_window加窗(对960的对称加窗)计算  并进行forward_transform存入X(481)
//对X计算各频段的能量存入Ex
static void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
  
  int i;
  float x[WINDOW_SIZE];//960
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(x);
  forward_transform(X, x);
#if TRAINING
  for (i=lowpass;i<FREQ_SIZE;i++)
    X[i].r = X[i].i = 0;
#endif
  compute_band_energy(Ex, X);
}

  
static int compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in) {
  int i;
  float E = 0;
  float *ceps_0, *ceps_1, *ceps_2; //float 数组
  float spec_variability = 0;
  float Ly[NB_BANDS];   //22
  float p[WINDOW_SIZE]; //960
  float pitch_buf[PITCH_BUF_SIZE>>1];//1728/2=864
  int pitch_index; //不懂
  float gain;
  float *(pre[1]);
  float tmp[NB_BANDS];
  float follow, logMax;
  //输入in为480  X为481 Ex为22的能量数组
  frame_analysis(st, X, Ex, in);
  //st.pitch_buf每次往左平移480 右侧添加新的in的480个数据
  RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE], in, FRAME_SIZE);


  pre[0] = &st->pitch_buf[0];//复制数组指针
  
  pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
  pitch_search(pitch_buf+(PITCH_MAX_PERIOD>>1), pitch_buf, PITCH_FRAME_SIZE,
               PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD, &pitch_index);
  pitch_index = PITCH_MAX_PERIOD-pitch_index;

  gain = remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
          PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
  
  st->last_period = pitch_index;
  st->last_gain = gain;

  //从pitch_buf(刚刚左移拼接in的1728数组)中截取960长度 起点为:PITCH_MAX_PERIOD(768)-pitch_index 
  for (i=0;i<WINDOW_SIZE;i++)
    p[i] = st->pitch_buf[PITCH_BUF_SIZE-WINDOW_SIZE-pitch_index+i];
  //计算出的p 加窗 前向转换存入P 计算能量存入Ep
  apply_window(p);
  forward_transform(P, p);
  compute_band_energy(Ep, P);
  //计算两个的点乘能量
  compute_band_corr(Exp, X, P);
  //进行一定的归一化
  for (i=0;i<NB_BANDS;i++) Exp[i] = Exp[i]/sqrt(.001+Ex[i]*Ep[i]);
  //X P 相关系数
  dct(tmp, Exp);
  //取前六个给下面
  //6个相关系数------------------------------------------------------------------feature[34:40]
  for (i=0;i<NB_DELTA_CEPS;i++) features[NB_BANDS+2*NB_DELTA_CEPS+i] = tmp[i];
  features[NB_BANDS+2*NB_DELTA_CEPS] -= 1.3;
  features[NB_BANDS+2*NB_DELTA_CEPS+1] -= 0.9;
  //周期------------------------------------------------------------------------feature[40]
  features[NB_BANDS+3*NB_DELTA_CEPS] = .01*(pitch_index-300);



  logMax = -2;
  follow = -2;
  //
  for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2+Ex[i]);
    Ly[i] = MAX16(logMax-7, MAX16(follow-1.5, Ly[i]));
    logMax = MAX16(logMax, Ly[i]);
    follow = MAX16(follow-1.5, Ly[i]);
    E += Ex[i];
  }
  if (!TRAINING && E < 0.04) {
    /* If there's no audio, avoid messing up the state. */
    RNN_CLEAR(features, NB_FEATURES);
    return 1;
  }
  //22个子带的倒谱系数 雏形-------------------------------------------------------------------feature[:22]
  //对Ly 利用common.dct_table 加权求和放入features的前22个
  dct(features, Ly);
  features[0] -= 12;
  features[1] -= 4;
  //每次下标低于0就＋8 使其在长度为8的列表里循环  CEPS_MEM=8   st->cepstral_mem[8][22]
  //这三个都是长度为22 的列
  ceps_0 = st->cepstral_mem[st->memid];     //取第  st->memid 列
  ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM+st->memid-1] : st->cepstral_mem[st->memid-1];//取 -1 列
  ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM+st->memid-2] : st->cepstral_mem[st->memid-2];//取 -2 列
  //ceps_0 等于 features[:22]
  for (i=0;i<NB_BANDS;i++) ceps_0[i] = features[i];
  st->memid++;
  
  for (i=0;i<NB_DELTA_CEPS;i++) {
    //更新一下前六个features---------------------------------------------------------------
    features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
    //6个一阶差分-------------------------------------------------------------------------feature[22:28]
    features[NB_BANDS+i] = ceps_0[i] - ceps_2[i]; 
    //6个二阶差分-------------------------------------------------------------------------feature[28:34]
    features[NB_BANDS+NB_DELTA_CEPS+i] =  ceps_0[i] - 2*ceps_1[i] + ceps_2[i];
  }
  /* Spectral variability features. */
  if (st->memid == CEPS_MEM) st->memid = 0;

  //dist储存的是st.cepstral_mem的每两列之间的距离的和 (任意i!=j的两列)
  //最后的总距离乘1个系数作为非平稳度量
  for (i=0;i<CEPS_MEM;i++)
  {
    int j;
    float mindist = 1e15f;
    for (j=0;j<CEPS_MEM;j++)
    {
      int k;
      float dist=0;
      for (k=0;k<NB_BANDS;k++)
      {
        float tmp;
        tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
        dist += tmp*tmp;
      }
      if (j!=i)
        mindist = MIN32(mindist, dist);
    }
    spec_variability += mindist;
  }
  //非平稳度量---------------------------------------------------------------feature[41]
  features[NB_BANDS+3*NB_DELTA_CEPS+1] = spec_variability/CEPS_MEM-2.1;
  return TRAINING && E < 0.1;//返回是否为静音
}







static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y) {
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(x, y);
  apply_window(x);
  for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
  //复制后480个 到 st.synthesis_mem中给下一次调用使用
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}


//mem[0] = b[0]x[i] - a[0]x[i] - a[0]mem[0] + mem[1]
//mem[1] = b[1]x[i] - a[1]x[i] - a[1]mem[0]
//x[i] = x[i] + mem[0]
//参数中y为更新后的x数组
static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}


void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g) {
  int i;
  float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  for (i=0;i<NB_BANDS;i++) {
#if 0
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
    r[i] = MIN16(1, MAX16(0, r[i]));
#else
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
    r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
#endif
    r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
  }
  interp_band_gain(rf, r);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r += rf[i]*P[i].r;
    X[i].i += rf[i]*P[i].i;
  }
  float newE[NB_BANDS];
  compute_band_energy(newE, X);
  float norm[NB_BANDS];
  float normf[FREQ_SIZE]={0};
  for (i=0;i<NB_BANDS;i++) {
    norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
  }
  interp_band_gain(normf, norm);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= normf[i];
    X[i].i *= normf[i];
  }
}

//用在降噪文件中 还没看
float rnnoise_process_frame(DenoiseState *st, float *out, const float *in) {
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[WINDOW_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS];
  float gf[FREQ_SIZE]={1};
  float vad_prob = 0;
  int silence;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
  silence = compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);

  if (!silence) {
    compute_rnn(&st->rnn, g, &vad_prob, features);
    pitch_filter(X, P, Ex, Ep, Exp, g);
    for (i=0;i<NB_BANDS;i++) {
      float alpha = .6f;
      g[i] = MAX16(g[i], alpha*st->lastg[i]);
      st->lastg[i] = g[i];
    }
    interp_band_gain(gf, g);
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
      X[i].r *= gf[i];
      X[i].i *= gf[i];
    }
#endif
  }

  frame_synthesis(st, out, X);
  return vad_prob;
}

#if TRAINING

//返回-0.5~0.5的随机数
static float uni_rand() {
  
  return rand()/(double)RAND_MAX-.5;
}

//将a和b两个长为2的数组每个元素初始化为-0.375,0.375之间的随机数
static void rand_resp(float *a, float *b) {
  
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

int main(int argc, char **argv) {
  int i;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_noise[2] = {0};
  float b_noise[2] = {0};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_hp_x[2]={0};
  float mem_hp_n[2]={0};
  float mem_resp_x[2]={0};
  float mem_resp_n[2]={0};
  float x[FRAME_SIZE];
  float n[FRAME_SIZE];
  float xn[FRAME_SIZE];
  int vad_cnt=0;
  int gain_change_count=0;
  float speech_gain = 1, noise_gain = 1;
  FILE *f1, *f2;
  int maxCount;
  DenoiseState *st;
  DenoiseState *noise_state;
  DenoiseState *noisy;
  st = rnnoise_create(NULL);
  noise_state = rnnoise_create(NULL);
  noisy = rnnoise_create(NULL);
  if (argc!=4) {
    fprintf(stderr, "usage: %s <speech> <noise> <count>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[1], "r");
  f2 = fopen(argv[2], "r");
  maxCount = atoi(argv[3]);
  //作者为了去掉前面的静音片段
  for(i=0;i<150;i++) {
    short tmp[FRAME_SIZE];
    fread(tmp, sizeof(short), FRAME_SIZE, f2);
  }
  while (1) {
    kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], N[FREQ_SIZE], P[WINDOW_SIZE];//481或者960个结构  每个结构里有两个float数字
    float Ex[NB_BANDS], Ey[NB_BANDS], En[NB_BANDS], Ep[NB_BANDS];
    float Exp[NB_BANDS];
    float Ln[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    short tmp[FRAME_SIZE];
    float vad=0;
    float E=0;
    if (count==maxCount) break;
    if ((count%10000)==0) fprintf(stderr, "%d\r", count);


    //每2821轮随机生成场景参数
    if (++gain_change_count > 2821) {
      //-40~19 /20  ->   10^(-2,0.95) -> (0.01,8.91)
      speech_gain = pow(10., (-40+(rand()%60))/20.);
      //-30~19 /20  ->   10^(-1.5,0.95) -> (0.03,8.91)
      noise_gain = pow(10., (-30+(rand()%50))/20.);
      //随机10%的几率 将 noise_gain清零
      if (rand()%10==0) noise_gain = 0;
      noise_gain *= speech_gain; //(0.0003,79.4)
      //随机10%的几率 将 speech_gain清零
      if (rand()%10==0) speech_gain = 0;
      gain_change_count = 0;
      //初始化为(-0.375~0.375)之间的随机数
      rand_resp(a_noise, b_noise);
      rand_resp(a_sig, b_sig);
      // 481*3000/24000*50^(0~1之间的随机数次方)=60.125*(1~50)=(60.125,3006.25)
      lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
      //0到21循环
      //0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14,    16, 20, 24, 28, 34, 40, 48, 60, 78, 100
      //如果eband5ms * 4 >上面的lowpass
      //eband5ms > lowpass/4 属于(15.03125,751.5625) 
      for (i=0;i<NB_BANDS;i++) {
        if (eband5ms[i]<<FRAME_SIZE_SHIFT > lowpass) {
          //找到满足条件的第一个下标i
          band_lp = i;
          break;
        }
      }
    }
    
    
    
    //读取语音片段x 和 其平方和E(模长平方)
    if (speech_gain != 0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f1);
      if (feof(f1)) {
        rewind(f1);
        fread(tmp, sizeof(short), FRAME_SIZE, f1);
      }
      for (i=0;i<FRAME_SIZE;i++) x[i] = speech_gain*tmp[i];//x=(0.…… ~ 8.91)*读取的speech
      for (i=0;i<FRAME_SIZE;i++) E += tmp[i]*(float)tmp[i];//E 储存所有speech 的平方和
    } else {
      //如果speech_gain清零了  x数组为0 E为0
      for (i=0;i<FRAME_SIZE;i++) x[i] = 0;
      E = 0;
    }
    if (noise_gain!=0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f2);
      if (feof(f2)) {
        rewind(f2);
        fread(tmp, sizeof(short), FRAME_SIZE, f2);
      }
      for (i=0;i<FRAME_SIZE;i++) n[i] = noise_gain*tmp[i]; //n=(0.…… ~ 79.……) *读取的noise
    } else {
      //如果noise_gain清零了  n数组都为0
      for (i=0;i<FRAME_SIZE;i++) n[i] = 0;  
    }

    //二阶滤波器分别对x和n序列做两次操作 
    biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
    biquad(n, mem_hp_n, n, b_hp, a_hp, FRAME_SIZE);
    biquad(n, mem_resp_n, n, b_noise, a_noise, FRAME_SIZE);

    //将x和n对位相加
    for (i=0;i<FRAME_SIZE;i++) xn[i] = x[i] + n[i];
    
    if (E > 1e9f) {
      //能量太大 清零
      vad_cnt=0;
    } else if (E > 1e8f) {
      //较大减5
      vad_cnt -= 5;
    } else if (E > 1e7f) {
      //中等 加1
      vad_cnt++;
    } else {
      //较小 加2
      vad_cnt+=2;
    }
    if (vad_cnt < 0) vad_cnt = 0; // 负数矫正为0
    if (vad_cnt > 15) vad_cnt = 15; // 太大矫正为上限15
    // 到此 vad_cnt 属于(0,15)

    if (vad_cnt >= 10) vad = 0; // vad_cnt>10 vad=0
    else if (vad_cnt > 0) vad = 0.5f;//0< vad_cnt <10  vad =0.5
    else vad = 1.f;  //vad_cnt =0   vad=1

    //st的analysis_mem和x 拼接后 进行apply_window以及傅里叶变换后的到Y
    //x的数值存入 st的analysis_mem 以供下次循环使用
    //计算变换后的能量存入Ey 
    frame_analysis(st, Y, Ey, x);
    //同理
    frame_analysis(noise_state, N, En, n);

    //取噪声能量的log 存入Ln
    for (i=0;i<NB_BANDS;i++) Ln[i] = log10(1e-2+En[i]);
    //noisy为DenoiseState X为481个kiss的数组  P为960的kiss数组  EX EP EXP 都为22数组  feature为42数组 xn为人声与噪声对位相加的长480片段
    int silence = compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, xn);
    
    //g[22] 期望增益
    pitch_filter(X, P, Ex, Ep, Exp, g);
    //printf("%f %d\n", noisy->last_gain, noisy->last_period);
    for (i=0;i<NB_BANDS;i++) {
      g[i] = sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));
      //
      if (g[i] > 1) g[i] = 1;
      //如果静音 or  频段i > lowpass
      if (silence || i > band_lp) g[i] = -1;
      //如果噪声 和 人声都很小  
      if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;
      //如果vad 和 噪声增益 都为0
      if (vad==0 && noise_gain==0) g[i] = -1;
    }
    count++;
#if 1
    fwrite(features, sizeof(float), NB_FEATURES, stdout);//42
    fwrite(g, sizeof(float), NB_BANDS, stdout); //22
    fwrite(Ln, sizeof(float), NB_BANDS, stdout);//22
    fwrite(&vad, sizeof(float), 1, stdout);//1
#endif
  }
  fprintf(stderr, "matrix size: %d x %d\n", count, NB_FEATURES + 2*NB_BANDS + 1);
  fclose(f1);
  fclose(f2);
  return 0;
}

#endif
