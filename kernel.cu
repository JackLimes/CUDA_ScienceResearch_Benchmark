#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include<iostream>
using namespace std;




__global__ void agg(int *b, unsigned int size) {
		b[blockIdx.x] = b[blockIdx.x] + 1;
}

int main(void) {

	//Hailiang Zhang's device counting code
	int deviceCount, device;
	short threads = 0;
	struct cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	if (cudaResultCode != cudaSuccess)
		deviceCount = 0;
	/* machines with no GPUs can still report one emulation device */
	for (device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties(&properties, device);
		if (properties.major != 9999) /* 9999 means emulation only */
			if (device == 0)
			{
				printf("multiProcessorCount %d\n", properties.multiProcessorCount);
				printf("maxThreadsPerMultiProcessor %d\n", properties.maxThreadsPerMultiProcessor);
				
			}
	}
	threads = properties.multiProcessorCount;
	printf("%d parallel blocks (because your gpu can support it!)\n", threads);

	//talk about how N must for some reason be a constant value. Must be manually changed for each graphics card.
	const short N = threads;
	int b[100] = { 0 }; //make this array so big there's no way any gpu will go over it.
	int *d_b;
	int span;
	_int64 result;
	clock_t begin = clock();

	cudaSetDevice(0);

	// Allocate space for device copies of  b and begin
	cudaMalloc((void **)&d_b, N * sizeof(int));

	// Copy inputs to device
	cudaMemcpy(d_b, &b, N * sizeof(int), cudaMemcpyHostToDevice);
	printf("How long do you wish to run this test in milliseconds? (1 second = 1000 Milliseconds) "); cin >> span;
	printf("Running...\n");
	while(clock() - begin < span) {
		agg << <N, 1 >> > (d_b, N);
	}

	// Copy result back to host
	cudaMemcpy(&b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost);

	// Cleanup
	cudaFree(d_b);
		

	result = b[0] * N;
	printf("Result: %d per element, with %d threads in pararallel leads to a final score of...\n%d", b[0], N, result);
	printf("\n");
	printf("{");
	for (int i = 0; i < N -1; i++) {
		printf("%d, ", b[i]);
	}
	printf("%d}\n", b[N - 1]);
	cudaDeviceReset();
	system("pause");
	return 0;

}
