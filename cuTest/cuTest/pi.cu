
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cuda.h>

const int nThreads = 1024;

void errorCheck(cudaError_t code, const char *func, const char *fileName, const int line)
{
	if (code)
	{
		std::cerr << "CUDA error = " << (int)code << " in file: " <<
			fileName << " function: " << func << " on line: " << line << "\n";
		cudaDeviceReset();
		exit(1);
	}
}

#define cudaErrorCheck(arg) errorCheck( (arg), #arg, __FILE__, __LINE__ )


__global__ void initRandState(curandState *randState, unsigned long seed)
{
	int idx = threadIdx.x;
	curand_init(seed, idx, 0, &randState[idx]);
}

__global__ void calcPi(curandState *globalRandState, size_t *d_sampleArray, size_t samples)
{
	int idx = threadIdx.x;
	float x = 0, y = 0;
	size_t count = 0;
	size_t max = (samples * samples) / nThreads;
	curandState localState = globalRandState[idx];
	for (size_t i = 0; i < max; ++i)
	{
		x = curand_uniform(&localState);
		y = curand_uniform(&localState);
		if (x * x + y * y <= 1)
			count++;
	}
	globalRandState[idx] = localState;
	d_sampleArray[idx] = count;
}

int main()
{
	size_t *h_samples = new size_t[nThreads];
	size_t samples;

	std::cout << "No. of samples: " << std::endl;
	std::cin >> samples;	
	
	curandState *d_randState;
	size_t *d_sampleArray;

	cudaErrorCheck(cudaMalloc((void**)&d_randState, nThreads * sizeof(curandState)));
	cudaErrorCheck(cudaMalloc((void**)&d_sampleArray, nThreads * sizeof(size_t)));

	auto start = std::chrono::high_resolution_clock::now();
	initRandState <<<1, nThreads>>> (d_randState, time(NULL));
	calcPi <<<1, nThreads>>> (d_randState, d_sampleArray, samples);

	cudaErrorCheck(cudaDeviceSynchronize());
	cudaErrorCheck(cudaMemcpy(h_samples, d_sampleArray, nThreads * sizeof(size_t), cudaMemcpyDeviceToHost));

	size_t s = 0;
	for (int i = 0; i < nThreads; i++)
		s += h_samples[i];

	long double pi = (4 * s) / (long double)(samples * samples);
	auto stop = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	

	std::cout << std::setprecision(12) << "Approx. value of Pi is: " << pi << std::endl;
	std::cout << "\nTime taken is " << diff.count() << " milliseconds." << std::endl;

	cudaErrorCheck(cudaFree(d_randState));
	cudaErrorCheck(cudaFree(d_sampleArray));
	delete[] h_samples;
}