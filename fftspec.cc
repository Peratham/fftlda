#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "fftw3.h"
#include "util.h"
#include "tensor.h"
#include "hash.h"
#include "count_sketch.h"
#include "fft_wrapper.h"
#include "matlab_wrapper.h"
#include "fast_tensor_power_method.h"


int B = 20;
int b = 11;
int K = 6;
int dim = 200;
int rank = 10;
double nsr = 0.01;
double rate = 0.01;
int TR = 10;
int para_L, para_T;

double decay = 0.8;

int dim1 = 100;
int dim2 = 100;

int evaldim = 1000;
double lnr_sigma = 2.0;

double thr = 0.1;

Tensor* T = new Tensor();
Hashes* hashes;
Hashes* asym_hashes[3];
FFT_wrapper *fft_wrapper, *ifft_wrapper;
Matlab_wrapper* mat_wrapper;

Tensor* gen_full_rank_tensor(int dim) {

	Tensor* ret = new Tensor(dim, TENSOR_STORE_TYPE_DENSE);
	for(int i = 0; i < dim; i++)
		for(int j = i; j < dim; j++)
			for(int k = j; k < dim; k++) {
				//double t = generate_std_normal();
				double t = (double)rand() / RAND_MAX;
				ret->A[IND3D(i,j,k,dim)] = t;
				ret->A[IND3D(i,k,j,dim)] = t;
				ret->A[IND3D(j,i,k,dim)] = t;
				ret->A[IND3D(j,k,i,dim)] = t;
				ret->A[IND3D(k,i,j,dim)] = t;
				ret->A[IND3D(k,j,i,dim)] = t;
			}

	double norm = 0;
	for(int p = 0; p < dim*dim*dim; p++)
		norm += SQR(ret->A[p]);
	double scale = 1.0 / sqrt(norm);
	for(int p = 0; p < dim*dim*dim; p++)
		ret->A[p] *= scale;

	assert(ret->symmetric_check());
	assert(safe_compare(ret->sqr_fnorm(), 1) == 0);

	return ret;

}

Tensor* gen_low_rank_tensor(int dim, int rank, double decay, double nsr, bool orthogonal = true, bool use_log_normal = false) {

	Tensor* ret = new Tensor(dim, TENSOR_STORE_TYPE_DENSE);

	double** u = new double*[rank];
	double* lambda = new double[rank];
	double t = 1;

	for(int k = 0; k < rank; k++) {
		lambda[k] = t;
		// t /= decay
		t -= decay;
		u[k] = new double[dim];
		for(int i = 0; i < dim; i++)
			//u[i]= (double)rand() / RAND_MAX + i % 2 * 10;
			//u[k][i] = (double)rand() / RAND_MAX - 0.5;
			//u[k][i] = (double)rand() / RAND_MAX;
			if (!use_log_normal)
                generate_uniform_sphere_point(dim, u[k]);
            else {
                mat_wrapper->lognrnd(dim, 1, lnr_sigma, u[k]);
                double sum = 0;
                for(int i = 0; i < dim; i++)
                    sum += SQR(u[k][i]);
                double scale = 1.0 / sqrt(sum);
                for(int i = 0; i < dim; i++)
                    u[k][i] *= scale;
            }
	}
    if (orthogonal) {
        puts("Doing gram-schmidt process");
        gram_schmidt_process(rank, dim, u);
    }

	for(int i = 0; i < dim; i++)
		for(int j = 0; j < dim; j++)
			for(int k = 0; k < dim; k++)
				for(int r = 0; r < rank; r++)
					ret->A[IND3D(i,j,k,dim)] += lambda[r] * u[r][i] * u[r][j] * u[r][k];

	double fnorm = sqrt(ret->sqr_fnorm());
	double scale = 1.0 / fnorm;
	for(int p = 0; p < dim*dim*dim; p++)
		ret->A[p] *= scale;

	assert(ret->symmetric_check());
	assert(safe_compare(ret->sqr_fnorm(), 1.0) == 0);

	// add noise
	double sigma = sqrt(nsr / (dim*dim*dim));
	for(int i = 0; i < dim; i++)
		for(int j = i; j < dim; j++)
			for(int k = j; k < dim; k++) {
				ret->A[IND3D(i,j,k,dim)] += sigma * generate_std_normal();
				double t = ret->A[IND3D(i,j,k,dim)];
				ret->A[IND3D(i,k,j,dim)] = t;
				ret->A[IND3D(j,i,k,dim)] = t;
				ret->A[IND3D(j,k,i,dim)] = t;
				ret->A[IND3D(k,i,j,dim)] = t;
				ret->A[IND3D(k,j,i,dim)] = t;
			}

	assert(ret->symmetric_check());

    FILE* fcore = fopen("tensor.core.dat", "wb");
    assert(fcore);
    for(int k = 0; k < rank; k++)
        fwrite(u[k], sizeof(double), dim, fcore);
    fclose(fcore);
    
    for(int k = 0; k < rank; k++)
        delete[] u[k];
	delete[] u;
	return ret;

}

// decay <= 0: linear decay
// eigens: double dim x dim
Tensor* gen_high_rank_tensor(int dim, double decay, double nsr, double* eigens) {
    
	Tensor* ret = new Tensor(dim, TENSOR_STORE_TYPE_DENSE);

	for(int k = 0; k < dim; k++) {
        generate_uniform_sphere_point(dim, eigens + k * dim);
    }
	double t = 1;

    for(int k = 0; k < dim; k++) {
        t = (decay <= 0)? 1.0 / (k+1) : t * decay;
        double* u = eigens + k * dim;
        for(int i1 = 0; i1 < dim; i1++)
            for(int i2 = 0; i2 < dim; i2++)
                for(int i3 = 0; i3 < dim; i3++)
                    ret->A[IND3D(i1,i2,i3,dim)] += t * u[i1] * u[i2] * u[i3];
    }

	double fnorm = sqrt(ret->sqr_fnorm());
	double scale = 1.0 / fnorm;
	for(int p = 0; p < dim*dim*dim; p++)
		ret->A[p] *= scale;

	assert(ret->symmetric_check());
	assert(safe_compare(ret->sqr_fnorm(), 1.0) == 0);

	// add noise
	double sigma = sqrt(nsr / (dim*dim*dim));
	for(int i = 0; i < dim; i++)
		for(int j = i; j < dim; j++)
			for(int k = j; k < dim; k++) {
				ret->A[IND3D(i,j,k,dim)] += sigma * generate_std_normal();
				double t = ret->A[IND3D(i,j,k,dim)];
				ret->A[IND3D(i,k,j,dim)] = t;
				ret->A[IND3D(j,i,k,dim)] = t;
				ret->A[IND3D(j,k,i,dim)] = t;
				ret->A[IND3D(k,i,j,dim)] = t;
				ret->A[IND3D(k,j,i,dim)] = t;
			}

	assert(ret->symmetric_check());
	return ret;    
    
}

void task1() {

	T = gen_low_rank_tensor(dim, rank, 0, nsr, true, false);
	T->save("tensor.dat");

}

void task2() {
    
    double* eigens = new double[rank * rank];
    T = gen_high_rank_tensor(dim, -1, nsr, eigens);
    T->save("tensor.dat");
    
}

void task3(char* file1, char* file2) {
    
    // slow robust tensor power method
    
    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;
    printf("squared T->fnorm = %lf\n", (T->sqr_fnorm()));
    
    double* lambda = new double[rank];
    double** u = new double*[rank];
    for(int k = 0; k < rank; k++)
        u[k] = new double[dim];
    
    slow_tensor_power_method(T, dim, rank, para_L, para_T, lambda, u);
    printf("squared residue fnorm = %lf\n", (T->sqr_fnorm()));
    
    FILE* f = fopen(file2, "wb");
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&rank, sizeof(int), 1, f);
    fwrite(lambda, sizeof(double), rank, f);
    for(int k = 0; k < rank; k++)
        fwrite(u[k], sizeof(double), dim, f);
    fclose(f);
    
}

void task4(char* file1, char* file2) {
    
    // fast sketch-based robust tensor power method
    
    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;
    printf("squared T->fnorm = %lf\n", (T->sqr_fnorm()));
    
    fft_wrapper = new FFT_wrapper(POWER2(b), FFTW_FORWARD);
    ifft_wrapper = new FFT_wrapper(POWER2(b), FFTW_BACKWARD);
    
    double* lambda = new double[rank];
    double** u = new double*[rank];
    for(int k = 0; k < rank; k++)
        u[k] = new double[dim];
    
    hashes = new Hashes(B, b, dim, K);
    CountSketch* cs_T = new CountSketch(hashes);
    cs_T->set_tensor(T);
    
    fast_tensor_power_method(cs_T, dim, rank, para_L, para_T, fft_wrapper, ifft_wrapper, lambda, u);
    
    double sum = 0;
    for(int i1 = 0; i1 < dim; i1 ++)
        for(int i2 = 0; i2 < dim; i2 ++) 
            for(int i3 = 0; i3 < dim; i3 ++) {
                double t = T->A[IND3D(i1, i2, i3, dim)];
                for(int k = 0; k < rank; k++)
                    t -= lambda[k] * u[k][i1] * u[k][i2] * u[k][i3];
                sum += SQR(t);
            }
    printf("squared residue fnorm = %lf\n", (sum));
    
    FILE* f = fopen(file2, "wb");
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&rank, sizeof(int), 1, f);
    fwrite(lambda, sizeof(double), rank, f);
    for(int k = 0; k < rank; k++)
        fwrite(u[k], sizeof(double), dim, f);
    fclose(f);    
    
}

void evaluate_and_save(Tensor* T, int dim, int rank, double* lambda, double* A, double* B, double* C, char* filename) {
    
    double sum = 0;
    for(int i1 = 0; i1 < dim; i1 ++)
        for(int i2 = 0; i2 < dim; i2 ++)
            for(int i3 = 0; i3 < dim; i3 ++) {
                double t = T->A[IND3D(i1, i2, i3, dim)];
                for(int k = 0; k < rank; k++)
                    t -= lambda[k] * A[IND2D(k, i1, dim)] * B[IND2D(k, i2, dim)] * C[IND2D(k, i3, dim)];
                sum += SQR(t);
            }
    printf("squared residue = %lf\n", sum);
    
    FILE* f = fopen(filename, "wb");
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&rank, sizeof(int), 1, f);
    fwrite(lambda, sizeof(double), rank, f);
    for(int k = 0; k < rank; k++)
        fwrite(A + k * dim, sizeof(double), dim, f);
    for(int k = 0; k < rank; k++)
        fwrite(B + k * dim, sizeof(double), dim, f);
    for(int k = 0; k < rank; k++)
        fwrite(C + k * dim, sizeof(double), dim, f);    
    fclose(f);     
    
}

void task5(char* file1, char* file2) {
    
    // slow ALS
    
    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;
    printf("squared T->fnorm = %lf\n", (T->sqr_fnorm()));

    double* U = new double[dim * rank];
    double* UT = new double[rank * dim];
    mat_wrapper->svds(T->A, dim, rank, U, NULL, NULL);    
    for(int k = 0; k < rank; k++)
        for(int i = 0; i < dim; i++)
            UT[IND2D(k,i,dim)] = U[IND2D(i,k,rank)];
    
    double* lambda = new double[rank];
    double* A = new double[rank * dim];
    double* B = new double[rank * dim];
    double* C = new double[rank * dim];
    for(int k = 0; k < rank; k++) {
        memcpy(A + k * dim, UT + k * dim, sizeof(double) * dim);
        memcpy(B + k * dim, UT + k * dim, sizeof(double) * dim);
        memcpy(C + k * dim, UT + k * dim, sizeof(double) * dim);
    }
    
    slow_asym_ALS(T, dim, rank, para_T, mat_wrapper, lambda, A, B, C);
    
    evaluate_and_save(T, dim, rank, lambda, A, B, C, file2);
    
}

void task6(char* file1, char* file2) {
    
    // fast ALS
    
    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;
    printf("squared T->fnorm = %lf\n", (T->sqr_fnorm()));

    double* U = new double[dim * rank];
    double* UT = new double[rank * dim];
    mat_wrapper->svds(T->A, dim, rank, U, NULL, NULL);    
    for(int k = 0; k < rank; k++)
        for(int i = 0; i < dim; i++)
            UT[IND2D(k,i,dim)] = U[IND2D(i,k,rank)];
    
    double* lambda = new double[rank];
    double* AA = new double[rank * dim];
    double* BB = new double[rank * dim];
    double* CC = new double[rank * dim];
    for(int k = 0; k < rank; k++) {
        memcpy(AA + k * dim, UT + k * dim, sizeof(double) * dim);
        memcpy(BB + k * dim, UT + k * dim, sizeof(double) * dim);
        memcpy(CC + k * dim, UT + k * dim, sizeof(double) * dim);
    }
    
    fft_wrapper = new FFT_wrapper(POWER2(b), FFTW_FORWARD);
    ifft_wrapper = new FFT_wrapper(POWER2(b), FFTW_BACKWARD);
    for(int i = 0; i < 3; i++) {
        asym_hashes[i] = new Hashes(B, b, dim, 6);
        asym_hashes[i]->to_asymmetric_hash();
    }
    AsymCountSketch* cs_T = new AsymCountSketch(3, asym_hashes);
    cs_T->set_tensor(T);
    fast_asym_ALS(cs_T, dim, rank, para_T, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, AA, BB, CC);
    
    evaluate_and_save(T, dim, rank, lambda, AA, BB, CC, file2);
    
}

int main(int argc, char* argv[]) {

	srand(time(0));
    
    mat_wrapper = new Matlab_wrapper();

	assert(argc >= 2);

	if (strcmp(argv[1], "synth_lowrank") == 0) {
        
        dim = atoi(argv[2]);
        rank = atoi(argv[3]);
        nsr = atof(argv[4]);
        
		task1();
        
	}
    else if (strcmp(argv[1], "synth_highrank") == 0) {
        
        dim = atoi(argv[2]);
        nsr = atof(argv[3]);
        
        task2();
    }
    else if (strcmp(argv[1], "slow_rbp") == 0) {
        
        rank = atoi(argv[3]);
        para_L = atoi(argv[4]);
        para_T = atoi(argv[5]);
        
        task3(argv[2], argv[6]);
        
    }
    else if (strcmp(argv[1], "fast_rbp") == 0) {
        
        rank = atoi(argv[3]);
        para_L = atoi(argv[4]);
        para_T = atoi(argv[5]);
        B = atoi(argv[6]);
        b = atoi(argv[7]);
        
        task4(argv[2], argv[8]);
        
    }
    else if (strcmp(argv[1], "slow_als") == 0) {
        
        rank = atoi(argv[3]);
        para_T = atoi(argv[4]);
        task5(argv[2], argv[5]);
        
    }
    else if (strcmp(argv[1], "fast_als") == 0) {
        
        rank = atoi(argv[3]);
        para_T = atoi(argv[4]);
        B = atoi(argv[5]);
        b = atoi(argv[6]);
        
        task6(argv[2], argv[7]);
        
    }

	return 0;

}
