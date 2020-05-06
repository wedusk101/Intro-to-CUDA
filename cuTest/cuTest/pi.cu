
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <iostream>
#include <cuda.h>


__global__ void init_cuda_rand(curandState *stateX, curandState *stateY, unsigned long seed)
{
	int idx = threadIdx.x;
	curand_init(seed, idx, 0, stateX);
	curand_init(seed, idx, 0, stateY);
}

__global__ void getRandom(curandState *globalStateX, curandState *globalStateY, float *d_randArrayX, float *d_randArrayY)
{
	int idx = threadIdx.x;
	curandState localStateX = globalStateX[idx];
	curandState localStateY = globalStateY[idx];
	float randValX = curand_uniform(&localStateX);
	float randValY = curand_uniform(&localStateY);
	d_randArrayX[idx] = randValX;
	d_randArrayY[idx] = randValY;
	globalStateX[idx] = localStateX;
	globalStateY[idx] = localStateY;
}

__global__ void calcPi(int *d_sampleArray, float *d_randArrayX, float *d_randArrayY, int samples)
{
	int idx = threadIdx.x;
	float x = 0, y = 0;
	int count = 0;
	for (int i = 0; i < samples * samples; i++)
	{
		x = d_randArrayX[idx];
		y = d_randArrayY[idx];
		if (x * x + y * y <= 1)
			count++;
	}
	d_sampleArray[idx] = count;
}

int main()
{
	const int ARRAY_SIZE = 128;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	int h_samples[ARRAY_SIZE];
	int samples = 1000;
	
	curandState *d_randStatesX;
	curandState *d_randStatesY;
	float *d_randArrayX;
	float *d_randArrayY;
	int *d_sampleArray;

	cudaMalloc((void**)&d_randArrayX, ARRAY_SIZE * sizeof(float));
	cudaMalloc((void**)&d_randArrayY, ARRAY_SIZE * sizeof(float));
	cudaMalloc((void**)&d_randStatesX, ARRAY_SIZE * sizeof(curandState));
	cudaMalloc((void**)&d_randStatesY, ARRAY_SIZE * sizeof(curandState));
	cudaMalloc((void**)&d_sampleArray, ARRAY_BYTES);

	init_cuda_rand <<<1, ARRAY_SIZE>>> (d_randStatesX, d_randStatesY, time(NULL));
	getRandom <<<1, ARRAY_SIZE>>> (d_randStatesX, d_randStatesY, d_randArrayX, d_randArrayY);
	calcPi <<<1, ARRAY_SIZE>>> (d_sampleArray, d_randArrayX, d_randArrayY, samples);

	cudaMemcpy(h_samples, d_sampleArray, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	long s = 0;
	for (int i = 0; i < ARRAY_SIZE; i++)
		s += h_samples[i];

	cudaFree(d_randArrayX);
	cudaFree(d_randArrayY);
	cudaFree(d_randStatesX);
	cudaFree(d_randStatesY);
	cudaFree(d_sampleArray);

	std::cout << (4 * s) / (double)(ARRAY_SIZE * samples * samples) << std::endl;
}