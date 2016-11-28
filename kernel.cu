
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>






__global__ void agg(int *b, unsigned int size) {
	clock_t begin = clock();
	while (begin - clock() < 1000) {
		b[blockIdx.x] = b[blockIdx.x] + 1;
	}
}

int main(void) {
	printf("Running...");

	//Hailiang Zhang's device counting code
	int deviceCount, device;
	int gpuDeviceCount = 0;
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
	printf("%d", threads);
	//talk about how N must for some reason be a constant value. Must be manually changed for each graphics card.
	const short N = 15;
	int b[N] = { 0 };
	int *d_b;
	cudaSetDevice(0);

	//while (clock() - begin < 1000){
		// Allocate space for device copies of  b
		cudaMalloc((void **)&d_b, N * sizeof(int));

		// Copy inputs to device
		cudaMemcpy(d_b, &b, N * sizeof(int), cudaMemcpyHostToDevice);

		agg <<<N, 1 >>> (d_b, N);

		// Copy result back to host
		cudaMemcpy(&b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost);

		// Cleanup
		cudaFree(d_b);

		
		
		//getchar();
	//}


	printf("Result: %d per element, with %d threads in pararallel leads to a final score of...\n%d", b[0], N, (b[0]* N));
	getchar();
	cudaDeviceReset();
	return 0;

}
