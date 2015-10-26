#ifndef FAST_TENSOR_POWER_METHOD_H_
#define FAST_TENSOR_POWER_METHOD_H_

#include "tensor.h"
#include "count_sketch.h"
#include "fft_wrapper.h"
#include "matlab_wrapper.h"

double fast_Tuuu(CountSketch* cs_T, CountSketch* f_cs_u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper);
void fast_TIuu(CountSketch* f_cs_T, CountSketch* f_cs_u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* ret);
void fast_TIuv(AsymCountSketch* f_cs_T, AsymCountSketch* f_cs_u2, AsymCountSketch* f_cs_u3, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* ret);
double fast_Tuvw(AsymCountSketch* f_cs_T, AsymCountSketch* f_cs_u1, AsymCountSketch* f_cs_u2, AsymCountSketch* f_cs_u3, int, FFT_wrapper* , FFT_wrapper* );
double fast_asym_Tuuu(CountSketch* f_cs_T, CountSketch* cs_u, CountSketch* cs_uu, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper);
double fast_collide_Tuuu(CountSketch* cs_T, CountSketch* f_cs_T, CountSketch* cs_u, CountSketch* cs_uu, double* u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper);
void fast_collide_TIuu(CountSketch* cs_T, CountSketch* f_cs_T, CountSketch* cs_u, CountSketch* cs_uu, double* u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* ret);

double fast_sqr_fnorm(AsymCountSketch* cs_T);
double fast_sqr_fnorm(CountSketch* cs_T);

// lambda: double[rank]
// v: double[rank][dim]
// Note: after factorization cs_T and tensor will store the deflated tensor

// use both Tuuu and TIuu
void fast_tensor_power_method(CountSketch* cs_T, int dim, int rank, int L, int T, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double** v);
double fast_collide_tensor_power_method(CountSketch* cs_T, int dim, int rank, int L, int T, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double** v, bool report_residue = false);
// W: double[dim1][dim2]
// v: double[k][dim2]
void fast_kernel_tensor_power_method(CountSketch* cs_T, int dim1, int dim2, double* W, int rank, int L, int T, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, Matlab_wrapper* mat_wrapper, double* lambda, double** v);

void slow_tensor_power_method(Tensor* tensor, int dim, int rank, int L, int T, double* lambda, double** v);

void slow_kernel_tensor_power_method(Tensor* tensor, int dim1, int dim2, double* W, int rank, int L, int T, Matlab_wrapper* mat_wrapper, double* lambda, double** v);

// A: rank x dim
// Note:  in the following two processes, assume A has already been initialized
void slow_ALS(Tensor* tensor, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, double* lambda, double* A);
void fast_ALS(CountSketch* cs_T, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double* A);

void slow_asym_ALS(Tensor* tensor, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, double* lambda, double* A, double* B, double* C);
// If report_residue is set to true: return ||T-...||_F^2 / ||T||^2. might take a lot of time
double fast_asym_ALS(AsymCountSketch* cs_T, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double* A, double* B, double* C, bool report_residue = false);
double fast_asym_tensor_power_method(AsymCountSketch* cs_T, int dim, int rank, int L, int T, Matlab_wrapper*, FFT_wrapper*, FFT_wrapper*, double* lambda, double** v, bool report_residue = false);

#endif
