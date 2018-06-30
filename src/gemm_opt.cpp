/*
 * gemm_opt.cpp
 *
 *  Created on: 2017/10/19
 *      Author: ZhangHua
 */

#include <iostream>
#include <csignal>
#include <fstream>

#ifdef CUBLAS_ENABLE
#undef __cdecl
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifndef BLAS_LIB_DISABLE
#include <clblast.h>
#include <clBLAS.h>
#endif

#ifdef MIOPENGEMM_ENABLE
#include <miopengemm/gemm.hpp>
#endif

#ifdef MKL_ENABLE
#include <mkl.h>
#endif

#include <tensor.hpp>
#include <device_instance.hpp>

using namespace std;
using namespace clnet;

int m, n, k, no;
bool parallel;
float alpha = 0;
float beta = 0;
Tensor *baseline = nullptr, *clBLAS = nullptr, *clBLAST = nullptr, *cublas = nullptr, *MKL = nullptr, *MIOpen = nullptr, *revised = nullptr, *clnet_tensor = nullptr;
/*json format:
 * {"M":M,"N":N,"K":K,"step":step,"repeat":repeat,"alpha":alpha,"beta":beta,"parallel":parallel,"verify":verify,"device","GTX 1050Ti","OS":OS,"date":date,"targets":["clblas","cublas"],
 * "data":[{"M":m,"N":n,"K":k,"times":times,"base":[time,delta],"clblas":[time,delta],"cublas":[time,delta]},...]}
 * */
ofstream* json_file = nullptr;
char device_name[64];
vector<cl::Event> all_events;

float* bufA = 0;
float* bufB = 0;
float* bufC = 0;
#ifdef CUBLAS_ENABLE
cublasHandle_t handle;
#endif

struct MyInitializer : Tensor, type::Structured {
	virtual void run(DeviceInstance& I) override {
		for (auto tensor : peers) {
			int i = 1;
			for (float *p = tensor->pointer, *end = p + tensor->volume; p < end; p++)
				*p = (float) i;//++;
		}
		std::vector<cl::Event> no_precondictions;
		for (auto tensor : peers) {
			memcpy(I.pointers[tensor], tensor->pointer, tensor->size);
			tensor->download(I, &no_precondictions);
		}
	}
	virtual std::vector<Tensor*> auxiliaries() override {
		return peers;
	}
};

T gemm_opt()
{
	T initializer = XavierNormalDistributionInitializer({}, 0, 2.34f);
//	T initializer = *new MyInitializer;
	int M = optional<int>("M", 2048); //dim_hidden
	int N = optional<int>("N", 512); //batch_size
	int K = optional<int>("K", 2048); //dim_in
	int B = optional<int>("bottom", 32); //bottom
	int STEP = optional<int>("step", 4);

	alpha = optional<double>("alpha", 1.0);
	beta = optional<double>("beta", 0);
	parallel = optional<int>("parallel", false);

	double REPEAT = optional<double>("repeat", 1.0);
	bool verify = optional<int>("verify", false);
	bool debug = optional<int>("debug", false);

	T w = Weight({K, M}, "w", &initializer);
	T x = Data({N, K}, &initializer, "x");
	T result = Data({M, N}, nullptr, "gemm");
	T graph = *new InstantTensor("gemm_loop_tester", {&x, &w},
		[M, N, K, B, STEP, REPEAT, &x, &w, &result, &initializer, verify, debug](InstantTensor* self, DeviceInstance& I) {
#ifdef CUBLAS_ENABLE
		extern void prepare_cublas(float* A, float* B, float* C, int K, int M, int N);
		if (cublas)
			prepare_cublas(w.pointer, x.pointer, result.pointer, K, M, N);
#endif

		int i = 0;
		no = 0;
		for (m = M; m >= B; m /= STEP)
			for (n = N; n >= B; n /= STEP)
				for (k = K; k >= B; k /= STEP) {
					int64 total = debug? 1 : (sqrt(REPEAT * M * N * K / m / n / k) - 0.8) * 50;
					logger << "M=" << m << " \tN=" << n << " \tK=" << k << " \ttimes=" << total << flush;
					if (json_file != nullptr) {
						if (i++ > 0)
							*json_file << ",\n";
						*json_file << "{\"M\":" << m << ",\"N\":" << n << ",\"K\":" << k << ",\"times\":" << total;
					}

					size_t minimum = INT_MAX, baseline_time = 0;
					size_t min_no = 0;
					for (size_t l = 0; l < self->peers.size(); l++) {
						auto tensor = self->peers[l];
						size_t time;
						try {
							logger << " \t" << tensor->alias << "=";
							if (json_file != nullptr)
								*json_file << ",\""<<  tensor->alias << "\":[";
							if (verify) {
								memset(I.pointers[&result], 0, result.size);
								result.download(I); //clear memory for preventing incorrect result report
							}
							//warm up
							tensor->run(I);
							wait_for_all_kernels_finished(I);
							all_events.clear();

							//timing the standard version now......
							time = MICROS(0);
							for (int j = 0; j < total; j++)
								tensor->run(I);
							if (parallel) {
#ifdef CUBLAS_ENABLE
								if (tensor == cublas)
									cudaDeviceSynchronize();
								else
#endif
								if (tensor != MKL)
									for (auto& event : all_events)
										event.wait();
							}
							time = MICROS(time);
							//end timing-----------------------------

							float each = time / 1000.0f / total;
							if (tensor == baseline)
								baseline_time = time;
							logger << time / 1000.0f << "/" << each;
							if (json_file != nullptr)
								*json_file << time / 1000.0f;
							if (minimum > time) {
								minimum = time;
								min_no = l;
							}
							if (verify) {
								float delta = 0;
#ifdef CUBLAS_ENABLE
								if (tensor == cublas)
									cudaMemcpy((void*)I.pointers[&result], (void*)bufC, result.size, cudaMemcpyDeviceToHost);
								else
#endif
								if (tensor == clnet_tensor) {
									clnet_tensor->peers[no]->upload(I);
									memcpy(I.pointers[&result], I.pointers[clnet_tensor->peers[no]], clnet_tensor->peers[no]->size);
								}
								else if (tensor != MKL) //For MKL, data has stored in I.pointers[&result]
									result.upload(I);
//								operate_tensor_data<float>(&result, I, {0, 0}, result.dimensions, result.dimensions, "1"); //check the value details
								if (tensor == baseline)
									memcpy(result.pointer, I.pointers[&result], result.size);
								else {
									auto p1 = result.pointer, p2 = I.pointers[&result];
									for (int j = 0; j < m * n; j ++, p1++, p2++) { //validiate the correction
										float diff = *p1 - *p2;
										delta += diff * diff;
									}
									logger << "(delta=" << delta << ")";
									if (json_file != nullptr)
										*json_file << "," << delta;
								}
							}
							if (tensor != baseline)
								logger << "/" << 1.0f * baseline_time / time;
						}
						catch (cl::Error& e) {
							logger << (debug? string(e.what()) + "(" + clErrorCodeDescriptions[-e.err()] + ")" : "error");
							if (json_file != nullptr)
								*json_file << "\"error\"";
						}
						catch (runtime_error& e) {
							logger << (debug? e.what() : "error");
							if (json_file != nullptr)
								*json_file << "\"error\"";
						}
						logger << flush;
						if (json_file != nullptr)
							*json_file << "]";
					}
					if (self->peers.size() > 1)
						logger << " \tWin=" << self->peers[min_no]->alias;
					if (json_file != nullptr)
						*json_file << ",\"win\":" << min_no << "}";
					logger << endl;
					no++;
				}

//-------------------------------------
//		const auto& context = I.queue.getInfo<CL_QUEUE_CONTEXT>();
//		cl::Image2D imageA(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_DEPTH, CL_FLOAT), M, K, 0);
//		cl::Image2D imageB(I.queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_ONLY, cl::ImageFormat(CL_DEPTH, CL_FLOAT), K, N, 0);
////		cl::Image2D imageC(I.queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_ONLY, cl::ImageFormat(CL_DEPTH, CL_FLOAT), M, N, 0);
//		auto& kernel = prepare_for_running_kernel(self, I);
//		kernel.setArg(0, I.buffers[&result]);
//		kernel.setArg(1, imageB);
//		kernel.setArg(2, imageA);
//		n = N; m = M; k = K;
//		kernel.setArg(3, n);
//		kernel.setArg(4, m);
//		kernel.setArg(5, k);
//		const auto& local = cl::NDRange(16, 16);
//		cl::NDRange global(m / 8, n / 8);
//		auto time = MICROS(0);
//		for (int r = 0; r < 9; r++) {
//			I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local/*cl::NullRange*/, &I.precondition_events, &I.events[&result]);
//			wait_for_all_kernels_finished(I);
//		}
//		time = MICROS(time);
//		logger << "Image:" << time << endl;
//-------------------------------------

#ifdef CUBLAS_ENABLE
		extern void shutdown_cublas();
		if (cublas != nullptr)
			shutdown_cublas();
#endif
#ifndef BLAS_LIB_DISABLE
		if (clBLAS != nullptr)
			clblasTeardown();
#endif
		if (json_file != nullptr) {
//			json_file->unget(); //remove comma at tail
			*json_file << "\n],\"rows\":" << i << "}\n";
			json_file->close();
		}
	}, {},
	[](InstantTensor* self) -> Tensor* { return nullptr; },
	[](InstantTensor* self) -> std::vector<Tensor*> { return self->peers; }
	);

	if (optional<int>("base", true)) {
		baseline = new InstantTensor("base", {}, {}, [&x, &w, &result](InstantTensor* self, DeviceInstance& I) {
			auto& kernel = prepare_for_running_kernel(self, I);
			kernel.setArg(0, I.buffers[&result]);
			kernel.setArg(1, I.buffers[&x]);
			kernel.setArg(2, I.buffers[&w]);
			kernel.setArg(3, nullptr);
			kernel.setArg(4, m);
			kernel.setArg(5, k);
			int local_size = find_proper_local_size(k, I.work_group_size);
			if (local_size > m * n)
				local_size = m * n;
			const auto& local = I.work_group_size <= 256? cl::NDRange(16, 16) : cl::NDRange(32, 32);
			cl::NDRange global(m * n);
			I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local/*cl::NullRange*/, &I.precondition_events, &I.events[&result]);
			if (parallel)
				all_events.push_back(I.events[&result]);
			else
				wait_for_all_kernels_finished(I);
		}, [](InstantTensor* self, DeviceInstance& I) -> string {
			string content;
			read_file_content<char>(OpenCL.location + "src/base.cl", content);
			return content;
		});
		graph.peers.push_back(baseline);
	}

	if (optional<int>("clnet", false)) {
		clnet_tensor = new InstantTensor("clnet", {}, {}, [&result, &x, &w](InstantTensor* self, DeviceInstance& I) {
			auto output = self->peers[no];
			auto x_ = output->inputs[0]->inputs[0];
			auto w_ = output->inputs[0]->inputs[1];
			I.events[x_] = I.events[w_] = I.events[&x];
			output->inputs[0]->run(I);
			if (parallel)
				all_events.push_back(I.events[output]);
			else
				wait_for_all_kernels_finished(I);
		});
		no = 0;
		for (m = M; m >= B; m /= STEP)
			for (n = N; n >= B; n /= STEP)
				for (k = K; k >= B; k /= STEP) {
					T x_ = Reshape(x, {n, k});
					T w_ = Reshape(w, {k, m});
					T output = FullyConnectedLayer(x_, w_, nullptr, "", "clnet_FC");
					clnet_tensor->peers.push_back(&output);
					no++;
				}
		graph.peers.push_back(clnet_tensor);
	}

	if (optional<int>("revised", false)) {
		int GroupSizeM = optional<int>("GroupSizeM", 16);
		int GroupSizeN = optional<int>("GroupSizeN", 16);
		int TileK = optional<int>("TileK", 16);
		int LoadM = optional<int>("LoadM", 8);
		int LoadN = optional<int>("LoadN", 8);
		int WIDTH = optional<int>("WIDTH", 1);
		revised = new InstantTensor("revised", {}, {}, [debug, LoadM, LoadN, GroupSizeM, GroupSizeN, &x, &w, &result](InstantTensor* self, DeviceInstance& I) {
			auto& kernel = prepare_for_running_kernel(self, I);
			kernel.setArg(0, I.buffers[&result]);
			kernel.setArg(1, I.buffers[&x]);
			kernel.setArg(2, I.buffers[&w]);
//			kernel.setArg(3, nullptr);
			kernel.setArg(3, n);
			kernel.setArg(4, m);
			kernel.setArg(5, k);
//			int local_size = find_proper_local_size(k, I.work_group_size);
//			if (local_size > m * n)
//				local_size = m * n;
//			const auto& local = I.work_group_size <= 256? cl::NDRange(16, 16) : cl::NDRange(32, 32);
			const auto& local = cl::NDRange(GroupSizeM, GroupSizeN);
			cl::NDRange global(m / LoadM, n / LoadN);
			I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local/*cl::NullRange*/, &I.precondition_events, &I.events[&result]);
			if (parallel)
				all_events.push_back(I.events[&result]);
			else
				wait_for_all_kernels_finished(I);
			if (debug)
				operate_tensor_data<float>(&result, I, {0, 0}, {17, 17}, result.dimensions); //check the value details
		}, [LoadM, LoadN, GroupSizeM, GroupSizeN, TileK, WIDTH](InstantTensor* self, DeviceInstance& I) -> string {
			if (cl_build_options.find("-DGroupSizeM=") == string::npos)
				cl_build_options += " -DGroupSizeM=" + to_string(GroupSizeM) + " -DGroupSizeN=" + to_string(GroupSizeN) + " -DTileK=" + to_string(TileK) + " -DLoadM=" + to_string(LoadM)
				+ " -DLoadN=" + to_string(LoadN) + " -DWIDTH=" + to_string(WIDTH);
			string content;
			read_file_content<char>(OpenCL.location + "src/revised.cl", content);
//			replace_all(content, "dim_in", to_string(k));
			return content;
		});
//		auto kernel = new InstantTensor("revised_kernel", {}, {}, [](InstantTensor* self, DeviceInstance& I) {}, [LoadM, LoadN, GroupSizeM, GroupSizeN, TileK, WIDTH](InstantTensor* self, DeviceInstance& I) -> string {
//			if (cl_build_options.find("-DGroupSizeM=") == string::npos)
//				cl_build_options += " -DGroupSizeM=" + to_string(GroupSizeM) + " -DGroupSizeN=" + to_string(GroupSizeN) + " -DTileK=" + to_string(TileK) + " -DLoadM=" + to_string(LoadM)
//				+ " -DLoadN=" + to_string(LoadN) + " -DWIDTH=" + to_string(WIDTH);
//			string content;
//			read_file_content<char>(OpenCL.location + "src/revised.cl", content);
////			replace_all(content, "dim_in", to_string(k));
//			return content;
//		});
		graph.peers.push_back(revised);
	}

	if (optional<int>("clblast", false)) {
#ifndef BLAS_LIB_DISABLE
		clBLAST = new InstantTensor("clblast", {}, {}, [&x, &w, &result](InstantTensor* self, DeviceInstance& I) {
			auto status = clblast::Gemm<float>(clblast::Layout::kColMajor, clblast::Transpose::kNo, clblast::Transpose::kNo, m, n, k, alpha, I.buffers[&w](), 0, m, I.buffers[&x](), 0, k, beta, I.buffers[&result](), 0, m, &I.queue(), &I.events[&result]());
			if (status != clblast::StatusCode::kSuccess)
				throw runtime_error("clblast error: " + to_string((int) status));
			if (parallel)
				all_events.push_back(I.events[&result]);
			else
				wait_for_all_kernels_finished(I);
		});
		graph.peers.push_back(clBLAST);
#endif
	}

	if (optional<int>("clblas", false)) {
#ifndef BLAS_LIB_DISABLE
		auto err = clblasSetup();
		if (err != CL_SUCCESS)
			throw runtime_error("clBLAS error: " + to_string((int) err));
		clBLAS = new InstantTensor("clBLAS", {}, {}, [&x, &w, &result](InstantTensor* self, DeviceInstance& I) {
			auto err = clblasSgemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans, m, n, k, alpha, I.buffers[&w](), 0, m,
					I.buffers[&x](), 0, k, beta, I.buffers[&result](), 0, m, 1, &I.queue(), 0, NULL, &I.events[&result]());
			if (err != CL_SUCCESS)
				throw runtime_error("clBLAS error: " + to_string((int) err));
			if (parallel)
				all_events.push_back(I.events[&result]);
			else
				wait_for_all_kernels_finished(I);
		});
		graph.peers.push_back(clBLAS);
#endif
	}

	if (optional<int>("miopen", false)) {
#ifdef MIOPENGEMM_ENABLE
		MIOpen = new InstantTensor("MIOpen", {}, {}, [&x, &w, &result](InstantTensor* self, DeviceInstance& I) {
			auto stat = MIOpenGEMM::gemm0<float>(true, false, false, m, n, k, alpha, I.buffers[&w](), 0, m,
					I.buffers[&x](), 0, k, beta, I.buffers[&result](), 0, m, &I.queue(), 0, NULL, &I.events[&result]());
			if (!stat.success)
				throw runtime_error("MIOpenGemm error: " + to_string(stat.ID));
			if (parallel)
				all_events.push_back(I.events[&result]);
			else
				wait_for_all_kernels_finished(I);
		});
		graph.peers.push_back(MIOpen);
#endif
	}

	if (optional<int>("cublas", false)) {
#ifdef CUBLAS_ENABLE
		cublas = new InstantTensor("cublas", {}, {}, [&x, &w, &result](InstantTensor* self, DeviceInstance& I) {
			auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, bufA, m, bufB, k, &beta, bufC, m);
			if (status != CUBLAS_STATUS_SUCCESS)
				exit(1);
			if (!parallel)
				cudaDeviceSynchronize();
		});
		graph.peers.push_back(cublas);
#endif
	}

	if (optional<int>("mkl", false)) {
#ifdef MKL_ENABLE
		MKL = new InstantTensor("MKL", {}, {}, [&x, &w, &result](InstantTensor* self, DeviceInstance& I) {
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, I.pointers[&w], m, I.pointers[&x], k, beta, I.pointers[&result], m);
		});
		graph.peers.push_back(MKL);
#endif
	}

	logger << "Comparing " << graph.peers[0]->alias;
	for (size_t i = 1; i < graph.peers.size(); i++)
		logger << ", " << graph.peers[i]->alias;
	logger << ", Verification " << (verify? "enabled" : "disabled");
	logger << ", Parallel " << (parallel? "enabled" : "disabled");
	logger << endl;

	string file = optional<string>("json", "");
	if (!file.empty()) {
		json_file = new ofstream(file, fstream::out);
		char date[32];
		time_t seconds = MICROS(0) / 1000000;
		tm* day = localtime(&seconds);
		sprintf(date, "%04d-%02d-%02d", day->tm_year + 1900, day->tm_mon + 1, day->tm_mday);
#ifdef __WINNT__
#define OS_name "Windows"
#elif defined(__APPLE__) || defined(__MACOSX)
#define OS_name "MacOS"
#else
#define OS_name "Linux"
#endif
		*json_file << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K << ",\"step\":" << STEP << ",\"repeat\":" << REPEAT << ",\"alpha\":" << alpha << ",\"beta\":" << beta
				<< ",\"parallel\":" << parallel << ",\"verify\":" << verify << ",\"device\":\"" << device_name << "\",\"OS\":\"" << OS_name << "\",\"date\":\"" << date << "\",\n\"targets\":" << "[";
		*json_file << "\"" << graph.peers[0]->alias << "\"";
		for (size_t i = 1; i < graph.peers.size(); i++)
			*json_file << ",\"" << graph.peers[i]->alias << "\"";
		*json_file << "],\"data\":[\n";
	}

//	if (debug)
//		return *new InstantTensor("debug", {&graph}, vector<Tensor*>());
	return graph;
}

#ifdef CUBLAS_ENABLE
void prepare_cublas(float* A, float* B, float* C, int K, int M, int N)
{
	// cuBLAS configuration
	cublasStatus_t status;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
		exit(1);
	// Prepare CUDA memory objects
	cudaMalloc((void**)&bufA, M*K*sizeof(*A));
	cudaMalloc((void**)&bufB, K*N*sizeof(*B));
	cudaMalloc((void**)&bufC, M*N*sizeof(*C));
	// Copy matrices to the GPU (also C to erase the results of the previous run)
	cudaMemcpy((void*)bufA, (void*)A, M*K*sizeof(*A), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)bufB, (void*)B, K*N*sizeof(*B), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)bufC, (void*)C, M*N*sizeof(*C), cudaMemcpyHostToDevice);
}

void shutdown_cublas()
{
	cublasStatus_t status;
	// Free the GPU memory objects
	cudaFree(bufA);
	cudaFree(bufB);
	cudaFree(bufC);
	// Clean-up cuBLAS
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS)
		exit(1);
}
#endif

namespace clnet {
extern unordered_map<string, string> key_values;
extern Tensor* _breakpoint;
}

int main(int argc, char** argv)
{
	signal(SIGINT, [](int signal) {
		logger << "User breaks by Ctrl+C." << endl;
		CLNET_TENSOR_GLOBALS |= CLNET_STEP_INTO_MODE;
		for (auto& iter : DeviceInstance::ALL)
			wait_for_all_kernels_finished(iter.second);
		exit(1);
	});

	OpenCL.location = optional<string>("location", "/GIT/gemm_optimization/");
	bool use_debugger = false, stop_on_startup = false, list_devices = false, display_structure = false, console_output = true, log_to_file = false;
	vector<int> devices;
	for (int i = 1; i < argc; i++) {
		string param(argv[i]);
		if (param.empty())
			return 1;
		else if (param[0] == ':' && i + 1 < argc)
			key_values[param.substr(1)] = argv[++i];
		else if (param == "/p")
			CLNET_TENSOR_GLOBALS |= CLNET_PREDICT_ONLY;
		else if (param == "/d")
			use_debugger = true;
		else if (param == "/ss")
			stop_on_startup = true;
		else if (param == "/ld")
			list_devices = true;
		else if (param == "/ds")
			display_structure = true;
		else if (param == "/nf")
			CLNET_TENSOR_GLOBALS ^= CLNET_FEED_FORWARD_FUSION | CLNET_BACK_PROPAGATE_FUSION;
		else if (param == "/os")
			CLNET_TENSOR_GLOBALS |= CLNET_OPENCL_SHOW_SOURCE;
		else if (param == "/all")
			OpenCL.device_type = CL_DEVICE_TYPE_ALL;
		else if (param == "/nlogc")
			console_output = false;
		else if (param == "/logf")
			log_to_file = true;
		else if (param[0] == '/') {
			if ((param[1] == '[' && param[param.length() - 1] == ']') || (param[1] >= '0' && param[1] <= '9')) //Linux shell strips '[' and ']' in "/[1,2]"
				parse_dimensions<int>(param.substr(1), &devices);
			else
				cout << "Unknown option " << param << " ignored." << endl;
		}
		else
			key_values["model"] = param;
	}

	if (log_to_file) {
		logger += optional<string>("log_file", OpenCL.location + "clnet.log");
		for (auto p = argv, end = argv + argc; p < end; p++) {
			string param(*p);
			if (param.find(' ') != string::npos)
				param = "\"" + param + "\"";
			logger << param << " ";
		}
		logger << endl;
	}
	if (console_output)
		logger += cout;
	if (devices.empty())
		devices = {0};
	int device_master = optional<int>("master", devices[0]);
	int device_debugger = optional<int>("debugger", use_debugger? devices[0] : -1);

	auto& device = OpenCL.find_devices()[devices[0]];
	string name = device.getInfo<CL_DEVICE_NAME>();
	memcpy(device_name, name.c_str(), name.length() + 1); //remove NVIDIA \0 char
	T graph = gemm_opt();
	if (list_devices)
		OpenCL.print_device_info(cout);
	if (display_structure)
		OpenCL.print_tensor_structure(graph);
	if (stop_on_startup)
		_breakpoint = &graph;
	OpenCL.run(graph, devices, device_debugger, device_master);
	return 0;
}
