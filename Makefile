CXXFLAGS= 
SOURCES = $(wildcard *.cc)
HEADS = $(wildcard *.h)
OBJS := $(SOURCES:.cc=.o)

all: fftspec fftlda

%.o: %.cc $(HEADS)
	g++ -O2 -m64 -Wno-unused-result -Wno-write-strings $(CXXFLAGS) -I "E:\MATLAB\R2013a\extern\include" -c $<

fftspec: $(OBJS)
	g++ -O2 -m64 -L "E:\MATLAB\R2013a\bin\win64" -Wno-unused-result -Wno-write-strings -o fftspec fftspec.o fast_tensor_power_method.o matlab_wrapper.o count_sketch.o tensor.o hash.o util.o libfftw3-3.dll -llibeng -llibmat -llibmx

fftlda: $(OBJS)
	g++ -O2 -m64 -L "E:\MATLAB\R2013a\bin\win64" -Wno-unused-result -Wno-write-strings -o fftlda fftlda.o fast_tensor_power_method.o tensor_lda.o tensor_lda_multithread.o corpus.o LDA.o matlab_wrapper.o count_sketch.o tensor.o hash.o util.o libfftw3-3.dll -llibeng -llibmat -llibmx

test: matlab_wrapper.o util.o
	g++ -O2 -m64 -L "E:\MATLAB\R2013a\bin\win64" -Wno-unused-result -Wno-write-strings -o test matlab_wrapper.o util.o -llibeng -llibmat -llibmx 

clean:
	rm -f *.o fftspe testc
