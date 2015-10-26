#ifndef TENSOR_H_
#define TENSOR_H_

#define TENSOR_STORE_TYPE_NULL 0
#define TENSOR_STORE_TYPE_DENSE 1
#define TENSOR_STORE_TYPE_DENSE_DISTRIBUTED 2
#define TENSOR_STORE_TYPE_SPARSE 3
#define TENSOR_STORE_TYPE_LOW_RANK 4

#include "matlab_wrapper.h"

class Tensor {

public:
	double* A;
	int dim;
	int store_type;

	double* values;
	int* idx[3];
	int nnz_count;
	double rate;
    
    double* Lambda;
    double* U;
    int rank;

	Tensor();
	Tensor(int dim, int store_type);
    Tensor(int dim, int rank, int store_type);
	~Tensor();

	int symmetric_check();

	void load(char*, int store_type);
	void save(char*);
    
    void load_view(int, Matlab_wrapper*);
    Tensor* whiten(Matlab_wrapper*, double* W); // return: rank x rank x rank tensor, W: real dim x rank

	double Tuuu(double* u, bool symeval = false);
	void TIuu(double* u, double* ret, bool symeval = false);
    void TIuv(double* u, double* v, double* ret);
	double sqr_fnorm(bool symeval = false);

	void to_sparse_format();
	void sparsify(double rate);

	void add_rank_one_update(double lambda, double* u);

private:
	void clear();

};

#endif
