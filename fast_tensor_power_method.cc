#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "config.h"
#include "util.h"
#include "hash.h"
#include "count_sketch.h"
#include "fft_wrapper.h"
#include "matlab_wrapper.h"

double fast_Tuuu(CountSketch* cs_T, CountSketch *f_cs_u, int dim, FFT_wrapper *fft_wrapper, FFT_wrapper *ifft_wrapper) {

	int B = cs_T->h->B;
	int b = cs_T->h->b;
	
	double* values = new double[B];
	fftw_complex* tvec = new fftw_complex[POWER2(b)];
	fftw_complex t, sum;

	for(int d = 0; d < B; d++) {
		for(int i = 0; i < POWER2(b); i++) {
			tvec[i][0] = t[0] = f_cs_u->cs[d][i][0]; 
			tvec[i][1] = t[1] = f_cs_u->cs[d][i][1];
			complex_mult(tvec[i], t, tvec[i]);
			complex_mult(tvec[i], t, tvec[i]);
		}
		ifft_wrapper->fft(tvec, tvec);
		sum[0] = sum[1] = 0;
		for(int i = 0; i < POWER2(b); i++) {
			t[0] = cs_T->cs[d][i][0];
			t[1] = cs_T->cs[d][i][1];
			sum[0] += tvec[i][0] * t[0] + tvec[i][1] * t[1];
			sum[1] += tvec[i][1] * t[0] - tvec[i][0] * t[1];
		}
		values[d] = sum[0];
	}


	qsort(values, B, sizeof(double), compare_double);

	double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)-1]);


	delete[] tvec;
	delete[] values;


	return ret;

}

void fast_TIuu(CountSketch* f_cs_T, CountSketch* f_cs_u, int dim, FFT_wrapper *fft_wrapper, FFT_wrapper *ifft_wrapper, double* ret) {

	int B = f_cs_T->h->B;
	int b = f_cs_T->h->b;
	
	double* values = new double[dim*B];
	fftw_complex* tvec = new fftw_complex[POWER2(b)];
	fftw_complex t;

	for(int d = 0; d < B; d++) {
		// compute tvec
		for(int i = 0; i < POWER2(b); i++) {
			complex_assign(f_cs_T->cs[d][i], tvec[i]);
			complex_mult_conj(tvec[i], f_cs_u->cs[d][i], tvec[i]);
			complex_mult_conj(tvec[i], f_cs_u->cs[d][i], tvec[i]);
		}
		// inverse fft
		ifft_wrapper->fft(tvec, tvec);
		// reading off the elements
		for(int i = 0; i < dim; i++) {
			int ind = f_cs_T->h->H[d][i];
			int angle = f_cs_T->h->Sigma[d][i];
			values[IND2D(i, d, B)] = tvec[ind][0] * Hashes::Omega[angle][0] + tvec[ind][1] * Hashes::Omega[angle][1];
		}
	}

	for(int i = 0; i < dim; i++) {
		double* base = values + i * B;
		qsort(base, B, sizeof(double), compare_double);
		ret[i] = (B&1)? base[B>>1] : 0.5 * (base[B>>1] + base[(B>>1)-1]);
	}

	delete[] values;
	delete[] tvec;

}

double fast_collide_Tuuu(CountSketch* cs_T, CountSketch* f_cs_T, CountSketch* cs_u, CountSketch* cs_uu, double* u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper) {

    int B = cs_T->h->B;
    int b = cs_T->h->b;

    cs_u->fft(fft_wrapper);
    cs_uu->fft(fft_wrapper);
    double* values = new double[B];
    fftw_complex t;
    
    const double scale1 = 1.0 / 6;
    const double scale2 = 3.0 / 6;
    const double scale3 = 2.0 / 6;
    
    for(int d = 0; d < B; d++) {
        values[d] = 0;
        for(int i = 0; i < POWER2(b); i++) {
            complex_assign(f_cs_T->cs[d][i], t);
            complex_mult_conj(t, cs_u->cs[d][i], t);
            complex_mult_conj(t, cs_u->cs[d][i], t);
            complex_mult_conj(t, cs_u->cs[d][i], t);
            values[d] += scale1 * t[0];
            complex_assign(f_cs_T->cs[d][i], t);
            complex_mult_conj(t, cs_uu->cs[d][i], t);
            complex_mult_conj(t, cs_u->cs[d][i], t);
            values[d] += scale2 * t[0];                   
        }
        values[d] /= POWER2(b);
        for(int i = 0; i < dim; i++) {
            int ind = (3 * cs_T->h->H[d][i]) & MASK2(b);
            int offset = (3 * cs_T->h->Sigma[d][i]) & (HASH_OMEGA_PERIOD - 1);
            assert(0 <= offset && offset < 4);
            complex_mult_conj(cs_T->cs[d][ind], Hashes::Omega[offset], t);
            values[d] += scale3 * u[i] * u[i] * u[i] * t[0];
        }
    }
    
    double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)-1]);
    delete[] values;
    
    return ret;

}

void fast_collide_TIuu(CountSketch* cs_T, CountSketch* f_cs_T, CountSketch* cs_u, CountSketch* cs_uu, double* u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* ret) {

    int B = cs_T->h->B;
    int b = cs_T->h->b;
    
    cs_u->fft(fft_wrapper);
    cs_uu->fft(fft_wrapper);
	double* values = new double[dim*B];
	memset(values, 0, sizeof(double) * dim*B);
	fftw_complex* tvec = new fftw_complex[POWER2(b)];
	fftw_complex t;    
	
	const double scale1 = 1.0 / 6;
	const double scale2 = 1.0 / 3;
	const double scale3 = 1.0 / 3;

    for(int d = 0; d < B; d++) {
        for(int i = 0; i < POWER2(b); i++) {
            complex_assign(cs_u->cs[d][i], t);
            complex_mult(t, cs_u->cs[d][i], t);
            complex_add(t, cs_uu->cs[d][i], t);
            complex_mult_conj(f_cs_T->cs[d][i], t, tvec[i]);
        }
        ifft_wrapper->fft(tvec, tvec);
        for(int i = 0; i < dim; i++) {
            int ind = cs_T->h->H[d][i];
            int omega = cs_T->h->Sigma[d][i];
            complex_mult_conj(tvec[ind], Hashes::Omega[omega], t);
            values[IND2D(i,d,B)] += scale1 * t[0];
        }
        for(int i = 0; i < POWER2(b); i++) {
            complex_mult_conj(f_cs_T->cs[d][i], cs_u->cs[d][i], tvec[i]);
        }
        ifft_wrapper->fft(tvec, tvec);
        for(int i = 0; i < dim; i++) {
            int ind = (cs_T->h->H[d][i] << 1) & MASK2(b);
            int omega = (cs_T->h->Sigma[d][i] << 1) & (HASH_OMEGA_PERIOD - 1);
            complex_mult_conj(tvec[ind], Hashes::Omega[omega], t);
            values[IND2D(i,d,B)] += scale2 * u[i] * t[0];
        }
        for(int i = 0; i < dim; i++) {
            int ind = (cs_T->h->H[d][i] * 3) & MASK2(b);
            int omega = (cs_T->h->Sigma[d][i] * 3) & (HASH_OMEGA_PERIOD - 1);
            complex_mult_conj(cs_T->cs[d][ind], Hashes::Omega[omega], t);
            values[IND2D(i,d,B)] += scale3 * SQR(u[i]) * t[0];
        }
    }
    
	for(int i = 0; i < dim; i++) {
		double* base = values + i * B;
		qsort(base, B, sizeof(double), compare_double);
		ret[i] = (B&1)? base[B>>1] : 0.5 * (base[B>>1] + base[(B>>1)-1]);
	}

	delete[] values;
	delete[] tvec;

}

void fast_TIuv(AsymCountSketch* f_cs_T, AsymCountSketch* f_cs_u2, AsymCountSketch* f_cs_u3, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* ret) {

    int B = f_cs_T->hs[0]->B;
    int b = f_cs_T->hs[0]->b;
    
	double* values = new double[dim*B];
	fftw_complex* tvec = new fftw_complex[POWER2(b)];
	fftw_complex t;

	for(int d = 0; d < B; d++) {
		// compute tvec
		for(int i = 0; i < POWER2(b); i++) {
			complex_assign(f_cs_T->cs[d][i], tvec[i]);
			complex_mult_conj(tvec[i], f_cs_u2->cs[d][i], tvec[i]);
			complex_mult_conj(tvec[i], f_cs_u3->cs[d][i], tvec[i]);
		}
		// inverse fft
		ifft_wrapper->fft(tvec, tvec);
		// reading off the elements
		for(int i = 0; i < dim; i++) {
			int ind = f_cs_T->hs[0]->H[d][i];
			int sigma = f_cs_T->hs[0]->Sigma[d][i];
			values[IND2D(i, d, B)] = sigma * tvec[ind][0];
		}
	}

	for(int i = 0; i < dim; i++) {
		double* base = values + i * B;
		qsort(base, B, sizeof(double), compare_double);
		ret[i] = (B&1)? base[B>>1] : 0.5 * (base[B>>1] + base[(B>>1)-1]);
	}

	delete[] values;
	delete[] tvec;

}

double fast_Tuvw(AsymCountSketch* f_cs_T, AsymCountSketch* f_cs_u1, AsymCountSketch* f_cs_u2, AsymCountSketch* f_cs_u3, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper) {

    int B = f_cs_T->hs[0]->B;
    int b = f_cs_T->hs[0]->b;
    
    double* values = new double[B];
    fftw_complex t, t_sum;
    
    for(int d = 0; d < B; d++) {
        t_sum[0] = t_sum[1] = 0;
        for(int i = 0; i < POWER2(b); i++) {
            complex_assign(f_cs_T->cs[d][i], t);
            complex_mult_conj(t, f_cs_u1->cs[d][i], t);
            complex_mult_conj(t, f_cs_u2->cs[d][i], t);
            complex_mult_conj(t, f_cs_u3->cs[d][i], t);
            complex_add(t_sum, t, t_sum);
        }
        values[d] = t_sum[0] / POWER2(b);
    }
    
    double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)-1]);
    delete[] values;
    
    return ret;

}

double fast_sqr_fnorm(AsymCountSketch* cs_T) {

    assert(cs_T->order == 3);
    int B = cs_T->B;
    int b = cs_T->b;
    
    double* values = new double[B];
    for(int d = 0; d < B; d++) {
        values[d] = 0;
        for(int i = 0; i < POWER2(b); i++)
            values[d] += SQR(cs_T->cs[d][i][0]);
    }
    
    qsort(values, B, sizeof(double), compare_double);
    double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)+1]);
    
    delete[] values;
    return ret;

}

double fast_sqr_fnorm(CountSketch* cs_T) {

    int B = cs_T->B;
    int b = cs_T->b;
    
    double* values = new double[B];
    for(int d = 0; d < B; d++) {
        values[d] = 0;
        for(int i = 0; i < POWER2(b); i++) 
            values[d] += complex_sqrnorm(cs_T->cs[d][i]);
    }
    
    qsort(values, B, sizeof(double), compare_double);
    double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)+1]);
    
    delete[] values;
    return ret;

}

void fast_tensor_power_method(CountSketch *cs_T, int dim, int rank, int L, int T, FFT_wrapper *fft_wrapper, FFT_wrapper *ifft_wrapper, double *lambda, double **v) {

	CountSketch *f_cs_T = new CountSketch(cs_T->h);
	CountSketch *f_cs_u = new CountSketch(cs_T->h);
	double *u = new double[dim];

	puts("--- Start fast tensor power method ---");

	for(int k = 0; k < rank; k++) {

		printf("For rank %d: ", k);

		// create FFT of cs_T
		f_cs_T->copy_from(cs_T);
		f_cs_T->fft(fft_wrapper);

		memset(v[k], 0, sizeof(double) * dim);
		double max_value = -1e100;

		for(int tau = 0; tau < L; tau++) {

			putchar('.');
			fflush(stdout);

			// Draw u randomly from the unit sphere and create its FFT count sketch
			generate_uniform_sphere_point(dim, u);
			f_cs_u->set_vector(u, dim);
			f_cs_u->fft(fft_wrapper);

			for(int t = 0; t < T; t++) {
				fast_TIuu(f_cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper, u);
				double norm = 0;
				for(int i = 0; i < dim; i++)
					norm += SQR(u[i]);
				double scale = 1.0 / sqrt(norm);
				for(int i = 0; i < dim; i++)
					u[i] *= scale;
				f_cs_u->set_vector(u, dim);
				f_cs_u->fft(fft_wrapper);
			}

			// compute T(uuu) and update v[k]
			double value = fast_Tuuu(cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper);
			if (value > max_value) { max_value = value; memcpy(v[k], u, sizeof(double) * dim);}

		}

		puts("#");
		fflush(stdout);

		// Do another round of power update
		f_cs_u->set_vector(v[k], dim);
		f_cs_u->fft(fft_wrapper);
		for(int t = 0; t < T; t++) {
			fast_TIuu(f_cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper, u);
			double norm = 0;
			for(int i = 0; i < dim; i++)
				norm += SQR(u[i]);
			double scale = 1.0 / sqrt(norm);
			for(int i = 0; i < dim; i++)
				u[i] *= scale;
			f_cs_u->set_vector(u, dim);
			f_cs_u->fft(fft_wrapper);
		}
		memcpy(v[k], u, sizeof(double) * dim);

		// compute the eigenvalue
		lambda[k] = fast_Tuuu(cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper);
//		printf("lambda[%d] = %lf\n", k, lambda[k]);

		// compute the deflated tensor
		cs_T->add_rank_one_tensor(-lambda[k], v[k], dim, fft_wrapper, ifft_wrapper);

	}

	puts("Completed.");

	delete[] u;
	delete f_cs_u;
	delete f_cs_T;

}

double fast_collide_tensor_power_method(CountSketch* cs_T, int dim, int rank, int L, int T, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double** v, bool report_residue) {

	CountSketch *f_cs_T = new CountSketch(cs_T->h);
	CountSketch *cs_u = new CountSketch(cs_T->h);
	CountSketch *cs_uu = new CountSketch(cs_T->h);
	double *u = new double[dim];
	
	double t_fnorm = 1;
	double residue = 0;
	
	if (report_residue) {
	    t_fnorm = fast_sqr_fnorm(cs_T);
	    printf("Before tensor power method: fnorm = %lf\n", t_fnorm);
	}

	puts("--- Start fast collide tensor power method ---");

	for(int k = 0; k < rank; k++) {

		printf("For rank %d: ", k);

		// create FFT of cs_T
		f_cs_T->copy_from(cs_T);
		f_cs_T->fft(fft_wrapper);

		memset(v[k], 0, sizeof(double) * dim);
		double max_value = -1e100;

		for(int tau = 0; tau < L; tau++) {

			putchar('.');
			fflush(stdout);

			// Draw u randomly from the unit sphere and create its FFT count sketch
			generate_uniform_sphere_point(dim, u);
			cs_u->set_vector(u, dim, 1);
			cs_uu->set_vector(u, dim, 2);

			for(int t = 0; t < T; t++) {
				fast_collide_TIuu(cs_T, f_cs_T, cs_u, cs_uu, u, dim, fft_wrapper, ifft_wrapper, u);
				double norm = 0;
				for(int i = 0; i < dim; i++)
					norm += SQR(u[i]);
				double scale = 1.0 / sqrt(norm);
				for(int i = 0; i < dim; i++)
					u[i] *= scale;
                cs_u->set_vector(u, dim, 1);
                cs_uu->set_vector(u, dim, 2);
			}

			// compute T(uuu) and update v[k]
			double value = fast_collide_Tuuu(cs_T, f_cs_T, cs_u, cs_uu, u, dim, fft_wrapper, ifft_wrapper);
			if (value > max_value) { max_value = value; memcpy(v[k], u, sizeof(double) * dim);}

		}

		puts("#");
		fflush(stdout);

		// compute the eigenvalue
		lambda[k] = max_value;

		// compute the deflated tensor
		cs_T->add_rank_one_tensor(-lambda[k], v[k], dim, fft_wrapper, ifft_wrapper, false);

	}

	puts("Completed.");
	
	if (report_residue) {
	    residue = fast_sqr_fnorm(cs_T);
	    printf("After tensor power method: fnorm = %lf\n", residue);
	}

	delete[] u;
	delete f_cs_T;
	delete cs_u;
	delete cs_uu;
	
	return residue / t_fnorm;

}

void fast_kernel_tensor_power_method(CountSketch *cs_T, int dim1, int dim2, double* W, int rank, int L, int T, FFT_wrapper *fft_wrapper, FFT_wrapper *ifft_wrapper, Matlab_wrapper* mat_wrapper, double *lambda, double **v) {

	CountSketch *f_cs_T = new CountSketch(cs_T->h);
	CountSketch *f_cs_u = new CountSketch(cs_T->h);
	double *u = new double[dim2];
	double *wu = new double[dim1];

	puts("--- Start fast kernel tensor power method ---");

	for(int k = 0; k < rank; k++) {

		printf("For rank %d: ", k);

		// create FFT of cs_T
		f_cs_T->copy_from(cs_T);
		f_cs_T->fft(fft_wrapper);

		memset(v[k], 0, sizeof(double) * dim2);
		double max_value = -1e100;

		for(int tau = 0; tau < L; tau++) {

			putchar('.');
			fflush(stdout);

			// Draw u randomly from the unit sphere and create its FFT count sketch
			generate_uniform_sphere_point(dim2, u);
			mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu); // wu = W * u
			f_cs_u->set_vector(wu, dim1);
			f_cs_u->fft(fft_wrapper);

			for(int t = 0; t < T; t++) {
				fast_TIuu(f_cs_T, f_cs_u, dim1, fft_wrapper, ifft_wrapper, wu);
				mat_wrapper->multiply(W, wu, dim1, dim2, dim1, 1, dim2, 1, true, false, u); // u = W' * wu
				double norm = 0;
				for(int i = 0; i < dim2; i++)
					norm += SQR(u[i]);
				double scale = 1.0 / sqrt(norm);
				for(int i = 0; i < dim2; i++)
					u[i] *= scale;
				mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu); // wu = W * u
				f_cs_u->set_vector(wu, dim1);
				f_cs_u->fft(fft_wrapper);
			}

			// compute T(uuu) and update v[k]
			double value = fast_Tuuu(cs_T, f_cs_u, dim1, fft_wrapper, ifft_wrapper);
			if (value > max_value) { max_value = value; memcpy(v[k], u, sizeof(double) * dim2);}

		}

		puts("#");
		fflush(stdout);

		// Do another round of power update
		mat_wrapper->multiply(W, v[k], dim1, dim2, dim2, 1, dim1, 1, false, false, wu); // wu = W * u
		f_cs_u->set_vector(wu, dim1);
		f_cs_u->fft(fft_wrapper);
		for(int t = 0; t < T; t++) {
			fast_TIuu(f_cs_T, f_cs_u, dim1, fft_wrapper, ifft_wrapper, wu);
			mat_wrapper->multiply(W, wu, dim1, dim2, dim1, 1, dim2, 1, true, false, u); // u = W' * wu
			double norm = 0;
			for(int i = 0; i < dim2; i++)
				norm += SQR(u[i]);
			double scale = 1.0 / sqrt(norm);
			for(int i = 0; i < dim2; i++)
				u[i] *= scale;
			mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu); // wu = W * u
			f_cs_u->set_vector(wu, dim1);
			f_cs_u->fft(fft_wrapper);
		}
		memcpy(v[k], u, sizeof(double) * dim2);

		// compute the eigenvalue
		lambda[k] = fast_Tuuu(cs_T, f_cs_u, dim1, fft_wrapper, ifft_wrapper);
//		printf("lambda[%d] = %lf\n", k, lambda[k]);

		// compute the deflated tensor
		mat_wrapper->mldivide(W, v[k], dim1, dim2, true, wu); // wu = W' \ u
		cs_T->add_rank_one_tensor(-lambda[k], wu, dim1, fft_wrapper, ifft_wrapper);

	}

	puts("Completed.");

	delete[] u;
	delete[] wu;
	delete f_cs_u;
	delete f_cs_T;

}

void slow_tensor_power_method(Tensor* tensor, int dim, int rank, int L, int T, double* lambda, double** v) {

	double *u = new double[dim];
	double *w = new double[dim];

	puts("--- Start slow tensor power method ---");

	for(int k = 0; k < rank; k++) {

		printf("For rank %d: ", k);

		memset(v[k], 0, sizeof(double) * dim);
		double max_value = -1e100;
		for(int tau = 0; tau < L; tau++) {
			putchar('.');
			fflush(stdout);
			generate_uniform_sphere_point(dim, u);
			for(int t = 0; t < T; t++) {
				tensor->TIuu(u, w);
				double norm = 0;
				for(int i = 0; i < dim; i++) norm += SQR(w[i]);
				double scale = 1.0 / sqrt(norm);
				for(int i = 0; i < dim; i++) w[i] *= scale;
				memcpy(u, w, sizeof(double) * dim);
			}
			double value = tensor->Tuuu(u);
			if (value > max_value) {
				max_value = value;
				memcpy(v[k], u, sizeof(double) * dim);
			}
		}
		puts("#");
		fflush(stdout);

		// Do another T round of update
		for(int t = 0; t < T; t++) {
			tensor->TIuu(v[k], w);
			double norm = 0;
			for(int i = 0; i < dim; i++) norm += SQR(w[i]);
			double scale = 1.0 / sqrt(norm);
			for(int i = 0; i < dim; i++) w[i] *= scale;
			memcpy(v[k], w, sizeof(double) * dim);
		}

		// Compute the eigenvalue
		lambda[k] = tensor->Tuuu(v[k]);

		// Compute the deflated tensor
		tensor->add_rank_one_update(-lambda[k], v[k]);
	}

	puts("Completed.");

	delete[] u;
	delete[] w;

}

void slow_kernel_tensor_power_method(Tensor* tensor, int dim1, int dim2, double* W, int rank, int L, int T, Matlab_wrapper* mat_wrapper, double* lambda, double** v) {

    double* u = new double[dim2];
    double* wu = new double[dim1];
    double* wv = new double[dim1];
    
    puts("-- Start slow kernel tensor power method --");
    
    for(int k = 0; k < rank; k++) {
    
        printf("For rank %d: ", k);
        
        memset(v[k], 0, sizeof(double) * dim2);
        double max_value = 1e-100;
        for(int tau = 0; tau < L; tau++) {
            putchar('.');
            fflush(stdout);
            generate_uniform_sphere_point(dim2, u);
            mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu); // wu = W * u      
            for(int t = 0; t < T; t++) {
                tensor->TIuu(wu, wv);
                mat_wrapper->multiply(W, wv, dim1, dim2, dim1, 1, dim2, 1, true, false, u); 
                double norm = 0;
                for(int i = 0; i < dim2; i++) norm += SQR(u[i]);
                double scale = 1.0 / sqrt(norm);
                for(int i = 0; i < dim2; i++) u[i] *= scale;
                mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu);
            }
            double value = tensor->Tuuu(wu);
            if (value > max_value) {
                max_value = value;
                memcpy(v[k], u, sizeof(double) * dim2);
            }
        }
        puts("");
        
        // do another T rounds
        mat_wrapper->multiply(W, v[k], dim1, dim2, dim2, 1, dim1, 1, false, false, wu);
        for(int t = 0; t < T; t++) {
            tensor->TIuu(wu, wv);
            mat_wrapper->multiply(W, wv, dim1, dim2, dim1, 1, dim2, 1, true, false, u); 
            double norm = 0;
            for(int i = 0; i < dim2; i++) norm += SQR(u[i]);
            double scale = 1.0 / sqrt(norm);
            for(int i = 0; i < dim2; i++) u[i] *= scale;
            mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu);        
        }
        
        memcpy(v[k], u, sizeof(double) * dim2);
        lambda[k] = tensor->Tuuu(wu);
        
        mat_wrapper->mldivide(W, v[k], dim1, dim2, true, wu);
        for(int i1 = 0; i1 < dim1; i1++)
            for(int i2 = 0; i2 < dim1; i2++)
                for(int i3 = 0; i3 < dim1; i3++)
                    tensor->A[IND3D(i1,i2,i3,dim1)] -= lambda[k] * wu[i1] * wu[i2] * wu[i3];
        
    }
    
    delete[] u;
    delete[] wu;
    delete[] wv;

}

void slow_ALS(Tensor* tensor, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, double* lambda, double* A) {

    puts("Start slow ALS ...");

    double* XCB = new double[rank * dim];
    double* AA = new double[rank * rank];
    
    for(int t = 0; t < T; t++) {
        
        printf("ALS Round %d ...\n", t);
        
        for(int k = 0; k < rank; k++) {
            tensor->TIuu(A + k * dim, XCB + k * dim);
        }
        
        mat_wrapper->multiply(A, A, rank, dim, rank, dim, rank, rank, false, true, AA);
        for(int i = 0; i < rank * rank; i++)
            AA[i] *= AA[i];
        mat_wrapper->pinv(AA, rank, rank, AA);
        mat_wrapper->multiply(AA, XCB, rank, rank, rank, dim, rank, dim, true, false, A);
        
        // normalize each row of A
        for(int k = 0; k < rank; k++) {
            lambda[k] = sqrt(vector_sqr_fnorm(A + k * dim, dim));
            double scale = 1.0 / lambda[k];
            for(int i = 0; i < dim; i++)
                A[IND2D(k,i,dim)] *= scale;
        }
        
    }
    
    delete[] XCB;
    delete[] AA;
    
}

void slow_asym_ALS_update(Tensor* tensor, int dim, int rank, Matlab_wrapper* mat_wrapper, double* lambda, double* A, double* B, double* C, double* XCB, double* CC, double* BB) {

    for(int k = 0; k < rank; k++) {
        tensor->TIuv(B + k * dim, C + k * dim, XCB + k * dim);
    }   
    
    mat_wrapper->multiply(B, B, rank, dim, rank, dim, rank, rank, false, true, BB);
    mat_wrapper->multiply(C, C, rank, dim, rank, dim, rank, rank, false, true, CC);
    for(int p = 0; p < rank*rank; p++)
        BB[p] *= CC[p];
    mat_wrapper->pinv(BB, rank, rank, BB);
    mat_wrapper->multiply(BB, XCB, rank, rank, rank, dim, rank, dim, true, false, A);
    
    // normalize each row of A
    for(int k = 0; k < rank; k++) {
        lambda[k] = sqrt(vector_sqr_fnorm(A + k * dim, dim));
        double scale = 1.0 / lambda[k];
        for(int i = 0; i < dim; i++)
            A[IND2D(k,i,dim)] *= scale;
    }

}

void slow_asym_ALS(Tensor* tensor, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, double* lambda, double* A, double* B, double* C) {

    printf("Slow Asym ALS: dim = %d, rank = %d\n", dim, rank);
    
    double* XCB = new double[rank * dim];
    double* CC = new double[rank * rank];
    double* BB = new double[rank * rank];

    for(int t = 0; t < T; t++) {
        
        printf("Slow asym round %d ...\n", t);
        
        slow_asym_ALS_update(tensor, dim, rank, mat_wrapper, lambda, A, B, C, XCB, CC, BB);
        slow_asym_ALS_update(tensor, dim, rank, mat_wrapper, lambda, B, C, A, XCB, CC, BB);
        slow_asym_ALS_update(tensor, dim, rank, mat_wrapper, lambda, C, A, B, XCB, CC, BB);
        
    }
    
    delete[] XCB;
    delete[] CC;
    delete[] BB;

}

void fast_ALS(CountSketch* cs_T, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double* A) {

    puts("Start Fast ALS ...");
    
    double* XCB = new double[rank * dim];
    double* AA = new double[rank * rank];
    
    cs_T->fft(fft_wrapper);
    CountSketch *f_cs_u = new CountSketch(cs_T->h);
    
    for(int t = 0; t < T; t++) {
        
        printf("Fast ALS Round %d ...\n", t);
        
        for(int k = 0; k < rank; k++) {
            f_cs_u->set_vector(A + k * dim, dim);
            f_cs_u->fft(fft_wrapper);
            fast_TIuu(cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper, XCB + k * dim);
        }
        
        mat_wrapper->multiply(A, A, rank, dim, rank, dim, rank, rank, false, true, AA);
        for(int i = 0; i < rank * rank; i++)
            AA[i] *= AA[i];
        mat_wrapper->pinv(AA, rank, rank, AA);
        mat_wrapper->multiply(AA, XCB, rank, rank, rank, dim, rank, dim, true, false, A);
        
        // normalize each row of A
        for(int k = 0; k < rank; k++) {
            lambda[k] = sqrt(vector_sqr_fnorm(A + k * dim, dim));
            double scale = 1.0 / lambda[k];
            for(int i = 0; i < dim; i++)
                A[IND2D(k,i,dim)] *= scale;
        }        
        
    }
    
    cs_T->fft(ifft_wrapper);
    delete[] XCB;
    delete[] AA;
    delete f_cs_u;

}

void fast_asym_ALS_update(AsymCountSketch* f_cs_T, int dim, int rank, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, 
    double* lambda, double* A, double* B, double* C, double* XCB, double* CC, double* BB, AsymCountSketch* cs_u2, AsymCountSketch* cs_u3) {
    
    for(int k = 0; k < rank; k++) {
        cs_u2->set_vector(B + k * dim, dim);
        cs_u3->set_vector(C + k * dim, dim);
        cs_u2->fft(fft_wrapper);
        cs_u3->fft(fft_wrapper);
        fast_TIuv(f_cs_T, cs_u2, cs_u3, dim, fft_wrapper, ifft_wrapper, XCB + k * dim);
    }
    
    /*for(int k = 0; k < rank; k++) {
        for(int i = 0; i < dim; i++)
            printf("%lf ", XCB[IND2D(k,i,dim)]);
        puts("");
    } */  
    
    mat_wrapper->multiply(B, B, rank, dim, rank, dim, rank, rank, false, true, BB);
    mat_wrapper->multiply(C, C, rank, dim, rank, dim, rank, rank, false, true, CC);
    for(int p = 0; p < rank*rank; p++)
        BB[p] *= CC[p];
    mat_wrapper->pinv(BB, rank, rank, BB);
    mat_wrapper->multiply(BB, XCB, rank, rank, rank, dim, rank, dim, true, false, A);
    
    // normalize each row of A
    for(int k = 0; k < rank; k++) {
        lambda[k] = sqrt(vector_sqr_fnorm(A + k * dim, dim));
        double scale = 1.0 / lambda[k];
        for(int i = 0; i < dim; i++)
            A[IND2D(k,i,dim)] *= scale;
    }    
    
}

double fast_asym_ALS(AsymCountSketch* cs_T, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double* A, double* B, double* C, bool report_residue) {

    double residue = 0;
    double t_fnorm = 0;
    
    if (report_residue) {
        t_fnorm = fast_sqr_fnorm(cs_T);
        printf("tensor fnorm before ALS: %lf\n", t_fnorm);
    }

    printf("Start fast aysm ALS, dim = %d, rank = %d, B = %d, b = %d\n", dim, rank, cs_T->B, cs_T->b);
    assert(cs_T->order == 3);
    
    double* XCB = new double[rank * dim];
    double* CC = new double[rank * rank];
    double* BB = new double[rank * rank];
        
    AsymCountSketch* cs_u1 = new AsymCountSketch(cs_T->hs[0]);
    AsymCountSketch* cs_u2 = new AsymCountSketch(cs_T->hs[1]);
    AsymCountSketch* cs_u3 = new AsymCountSketch(cs_T->hs[2]);
    
    cs_T->fft(fft_wrapper);
    
    for(int t = 0; t < T; t++) {
    
        printf("Fast asym ALS round %d ...\n", t);
        
        fast_asym_ALS_update(cs_T, dim, rank, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, A, B, C, XCB, CC, BB, cs_u2, cs_u3);
        fast_asym_ALS_update(cs_T, dim, rank, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, B, C, A, XCB, CC, BB, cs_u2, cs_u3);
        fast_asym_ALS_update(cs_T, dim, rank, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, C, A, B, XCB, CC, BB, cs_u2, cs_u3);
        
    }
    
    cs_T->fft(ifft_wrapper);
    
    delete[] XCB;
    delete[] CC;
    delete[] BB;
    
    if (report_residue) {

        for(int k = 0; k < rank; k++) {
            cs_u1->set_vector(A + k * dim, dim); cs_u1->fft(fft_wrapper);
            cs_u2->set_vector(B + k * dim, dim); cs_u2->fft(fft_wrapper);
            cs_u3->set_vector(C + k * dim, dim); cs_u3->fft(fft_wrapper);
            cs_T->add_rank_one_tensor(-lambda[k], dim, cs_u1, cs_u2, cs_u3, fft_wrapper, ifft_wrapper);
        }
        residue = fast_sqr_fnorm(cs_T);
        printf("After ALS: residue = %lf\n", residue);
        
        delete cs_u1;
        delete cs_u2;
        delete cs_u3;        
        
        return residue / t_fnorm;
    }
    else {
    
        delete cs_u1;
        delete cs_u2;
        delete cs_u3;    
    
        return 0;
    }

}

double fast_asym_tensor_power_method(AsymCountSketch* cs_T, int dim, int rank, int L, int T, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double** v, bool report_residue = false) {

    double residue = 0;
    double t_fnorm = 0;
    
    double* best_u = new double[dim];
    double* u = new double[dim];
    AsymCountSketch* asym_u[3];
    for(int i = 0; i < 3; i++) {
        asym_u[i] = new AsymCountSketch(cs_T->hs[i]);
    }
    
    if (report_residue) {
        t_fnorm = fast_sqr_fnorm(cs_T);
        printf("tensor fnorm before tensor power method: %lf\n", t_fnorm);
    }
    
    cs_T->fft(fft_wrapper);
    
    for(int k = 0; k < rank; k++) {
    
        printf("Round %d: ", k);
    
        double best_value = -1e100;
        memset(best_u, 0, sizeof(double) * dim);
        for(int tau = 0; tau < L; tau++) {
            putchar('.');
            fflush(stdout);
            generate_uniform_sphere_point(dim, u);
            for(int t = 0; t < T; t++) {
                
                asym_u[1]->set_vector(u, dim); asym_u[1]->fft(fft_wrapper);
                asym_u[2]->set_vector(u, dim); asym_u[2]->fft(fft_wrapper);
                
                fast_TIuv(cs_T, asym_u[1], asym_u[2], dim, fft_wrapper, ifft_wrapper, u);
                
                double sum = 0;
                for(int i = 0; i < dim; i++)
                    sum += SQR(u[i]);
                double scale = 1.0 / sqrt(sum);
                for(int i = 0; i < dim; i++)
                    u[i] *= scale;                
                                   
            }
            
            for(int i = 0; i < 3; i++) {
                asym_u[i]->set_vector(u, dim);
                asym_u[i]->fft(fft_wrapper);
            }
            double value = fast_Tuvw(cs_T, asym_u[0], asym_u[1], asym_u[2], dim, fft_wrapper, ifft_wrapper);
            if (value > best_value) {
                best_value = value;
                memcpy(v[k], u, sizeof(double) * dim);
            }
        }
        
        memcpy(u, v[k], sizeof(double) * dim);
        for(int t = 0; t < T; t++) {
        
            asym_u[1]->set_vector(u, dim); asym_u[1]->fft(fft_wrapper);
            asym_u[2]->set_vector(u, dim); asym_u[2]->fft(fft_wrapper);
              
            fast_TIuv(cs_T, asym_u[1], asym_u[2], dim, fft_wrapper, ifft_wrapper, u);
                
            double sum = 0;
            for(int i = 0; i < dim; i++)
                sum += SQR(u[i]);
            double scale = 1.0 / sqrt(sum);
            for(int i = 0; i < dim; i++)
                u[i] *= scale;                
                                                       
        }
        
        for(int i = 0; i < 3; i++) {
            asym_u[i]->set_vector(u, dim);
            asym_u[i]->fft(fft_wrapper);
        }   
        lambda[k] = fast_Tuvw(cs_T, asym_u[0], asym_u[1], asym_u[2], dim, fft_wrapper, ifft_wrapper);
        memcpy(v[k], u, sizeof(double) * dim);
        
        // deflation
        cs_T->add_rank_one_tensor(-lambda[k], dim, asym_u[0], asym_u[1], asym_u[2], fft_wrapper, ifft_wrapper, false);
        
        printf("#\n");
    
    }
    
    cs_T->fft(ifft_wrapper);
    if (report_residue) {
        residue = fast_sqr_fnorm(cs_T);
        printf("tensor fnorm after tensor power method: %lf\n", residue);
        residue /= t_fnorm;
    }
    
    delete best_u;
    delete u;
    for(int i = 0; i < 3; i++)
        delete asym_u[i];
        
    return residue;

}
