all:
	mkdir -p ./build
	g++ -std=c++1y -m64 -DMKL_ENABLE -DCUBLAS_ENABLE -I./include -I/usr/local/cuda/include -I/opt/intel/mkl/include -O3 -Wall -c -fopenmp -o build/gemm_opt.o ./src/gemm_opt.cpp
	g++ -L/usr/local/cuda/lib64 -L./lib/linux -L./lib -o build/gemm_opt.exe build/gemm_opt.o -lOpenCLNet -lOpenCL -lgomp -lclblast -lcudart -lcublas -lclBLAS -lmkl_rt
	export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64_lin/
	if [ ! -f "/lib64/libclblast.so" ]; then (cp -f lib/libclblast.so /lib64/libclblast.so) fi
	if [ ! -f "/lib64/libclBLAS.so" ]; then (cp -f lib/libclBLAS.so /lib64/libclBLAS.so) fi

clean:
	rm ./build -R -f