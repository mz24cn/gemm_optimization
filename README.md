# gemm(matrix multiplication) optimization 矩阵乘法优化
The repository targets the gemm function performance optimization. It compares several libraries clBLAS, clBLAST, MIOpenGemm, Intel MKL(CPU) and cuBLAS(CUDA) on different matrix sizes/vendor's hardwares/OS. Out-of-the-box easy as MSVC, MinGW, Linux(CentOS) x86_64 binary provided.  
在不同矩阵大小/硬件/操作系统下比较几个BLAS库的sgemm函数性能，提供binary，开盒即用。  
# Some results 部分结果
[GPU device GTX1080 (4096~32) \* (4096~32) \* (4096~32) on Windows](https://mz24cn.github.io/gemm_optimization/results/html/results.html?GTX1080_4096_4096_4096.json)  
[GPU device GTX1050Ti (2048~32) \* (2048~32) \* (2048~32) on Windows](https://mz24cn.github.io/gemm_optimization/results/html/results.html?GTX1050Ti_2048_2048_2048.json)  
[GPU device R9 290X (2048~32) \* (2048~32) \* (2048~32) on Windows](https://mz24cn.github.io/gemm_optimization/results/html/results.html?Hawaii_2048_2048_2048.json)  
# How to Build
The repository contains an eclipse CDT project, a Microsoft Visual Studio VC project, and a Linux Makefile. Some package include file and binary library files are included. But it may be incomplete (for example, some Intel MKL runtime libraries for some CPU types). I think it is not difficult to solve such issues for the people who cares gemm optimization.
# How to Run
`.\gemm_optimization.exe /1 :clblast 1 :clblas 1 :cublas 1 :mkl 1 :verify 1 :json D:\GTX1050Ti_Windows.json :M 2048 :N 2048 :K 2048 :step 2`  
This command line indicates the gemm computing on OpenCL device no. 1, clblast, clblas, NVIDIA cublas, Intel MKL enabled, data correction verification enabled, output data as json file 'D:\GTX1050Ti_Windows.json', the matrix multiplication computing starts from size A\[2048\[2048] \* B\[2048]\[2048], each dimension step down with factor 2 (2048, 1024, 512, ..., etc.).