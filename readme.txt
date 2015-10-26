Developed by Yining Wang (http://www.yining-wang.com)
Please cite the following paper:
Fast and guaranteed tensor decomposition via sketching. arXiv:1506.04448.
By Yining Wang, Hsiao-Yu Tung, Alex Smola and Anima Anandkumar.

Required packages:
FFTW
Matlab (>=2013)

============================================================================================
Fast tensor decomposition

1. Generating a (noisy) low-rank tensor
./fftspec synth_lowrank [dim] [rank] [nsr]

[dim]: the ambient dimension of the tensor
[rank]: the intrinsic dimension of the tensor
[nsr]: noise-to-signal ratio.

The resulting tensor will be saved to "tensor.dat" file. See Dense tensor format below for the format of the saved tensor.

2. Generating a (noisy) high-rank tensor
./fftspec synth_highrank [dim] [nsr]

The resulting tensor would have eigenvalues linearly decaying. That is, lambda_i = 1.0 / i

3. Brute-force robust tensor power method
./fftspec slow_rbp [input tensor data file] [rank] [L] [T] [result file]

[input tensor data file]: the tensor to be decomposed. See below sections for formatting instructions.
[rank]: target rank
[L, T]: parameters for robust tensor power method. See paper for details.
[result file]: store the factorization of input tensor. 
Format: (binary file)
dim (int), rank (int)
lambda (double x rank)
u1 (double x dim) (the first eigenvector)
u2 (double x dim) (the second eigenvector)
...

4. Fast sketch-based robust tensor power method
./fftspec slow_rbp [input tensor data file] [rank] [L] [T] [B] [b] [result file]

[input tensor data file]: the tensor to be decomposed. See below sections for formatting instructions.
[rank]: target rank
[L, T]: parameters for robust tensor power method. See paper for details.
[B, b]: parameters for sketching based tensor decomposition. See papaer for details.
[result file]: store the factorization of input tensor. 

5. Brute-force Alternating Least Squares (ALS)
./fftspec slow_als [input tensor data file] [rank] [T] [result_file]

[input tensor data file]: the tensor to be decomposed. See below sections for formatting instructions.
[rank]: target rank
[T]: parameters for ALS. See paper for details.
[result file]: store the factorization of input tensor. 

Note that for ALS, we have three (possibly different) aspects A, B and C. The result file will store all of them, in the exact order.

6. Fast sketch sketch-based ALS
./fftspec fast_als [input tensor data file] [rank] [T] [B] [b] [result_file]

[input tensor data file]: the tensor to be decomposed. See below sections for formatting instructions.
[rank]: target rank
[T]: parameters for ALS. See paper for details.
[B, b]: parameters for sketching based tensor decomposition. See papaer for details.
[result file]: store the factorization of input tensor. 

============================================================================================
Fast spectral LDA

Before running, please first open the "fftlda.cc" file and set parameters for the LDA model to be learnt:
V: number of vocabularies
K: number of topics
B, b: sketch parameters. See paper for details.
L, T: robust tensor power method parameters. See paper for details.
num_threads: number of threads to use
alpha0: parameter for spectral LDA (see paper for details). **strongly recommended to set as 1**

1. Generating synthetic corpus: 
./fftlda synth [D] [m]

D: number of documents
m: number of words per documents

The synthesized corpus and model will be stored in "synth.corpus" and "synth.model"

For formats of corpus and model files, see below sections.

2. Running (and evaluating) topic models
./fftlda fast_speclda [corpus file] [model file] 
or 
./fftlda fast_speclda [corpus file] [model file] [reference model file]

corpus file: file name for the corpus
model file: file name to store the fitted model
reference model file: OPTIONAL. If provided, the program will try to evaluate the model trained with respect to the provided reference model file. This is not the only way to evaluate the obtained LDA model. One can always evaluate an obtained LDA model by its perplexity (see next) without any knowledge of a reference model.

3. Evaluating perplexity
./fftlda eval [model file] [test corpus file] [result file]

model file: the model to be evaluated
test corpus file: evaluate the perplexity of given model on this particular test corpus
result file: store evaluation results

WARNING: evaluation is extremely slow! Make sure your test corpus has at most 1000 documents.

4. Producing topic assignments
./fftlda gibbs [model file] [corpus file] [topic assignment file]

[model file]: the trained spectral LDA model
[corpus file]: the training corpus
[topic assignment file]: topic assignments for every word in the training corpus

Perform Gibbs sampling to produce a topic assignment. Could then be used as initialization of a collapsed Gibbs sampler and obtain much more accurate results.

============================================================================================
Dense tensor format (*.dat, binary file):
dim (int)
data (double x dim*dim*dim)

NOTE: ALL WORDS MUST BE LISTED IN ASCENDING ORDER W.R.T. WORD ID
Corpus format (*.corpus, binary file)
# of documents (int)
	doc_id1 (long long); # of items (int); word ids (int x # of items); word occurrences (int x # of items)
    doc_id2 (long long); # of items (int); word ids (int x # of items); word occurrences (int x # of items)
    doc_id3 (long long); # of items (int); word ids (int x # of items); word occurrences (int x # of items)
    ...


LDA model format (*.model, binary file)
V (int), K (int)
alpha (double x K)
beta (double x V)
phi_1 (double x V)
phi_2 (double x V)
...
phi_K (double x V)

Gibbs sampling data format
# of topics (int)
# of documents (int)
	doc_id (long long); h (double x K); z (int x K)

Tensor power method pre-process data format
# of documents (int)
K (int)
W * n_1 (double * K) # of words (int)
W * n_2 (double * K) # of words (int)
...
W * n_d (double * K) # of words (int)
