#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <iostream>
#include <cuda.h>


__global__ void square(int *d_in, int *d_out)
{
	int idx = threadIdx.x;
	int x = d_in[idx];
	d_out[idx] = x * x;
}

int main()
{
	const int ARRAY_SIZE = 128;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	int h_in[ARRAY_SIZE];
	int h_out[ARRAY_SIZE];
	int *d_in;
	int *d_out;

	cudaMalloc((void**)&d_in, ARRAY_BYTES);
	cudaMalloc((void**)&d_out, ARRAY_BYTES);

	for (int i = 0; i < ARRAY_SIZE; i++)
		h_in[i] = i;

	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	square <<<1, ARRAY_SIZE >>> (d_in, d_out);

	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

	for (int i = 0; i < ARRAY_SIZE; i++)
		std::cout << h_out[i] << std::endl;
}