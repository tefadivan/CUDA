#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>

#define MAX_SHIFT 2
#define BLOCK_SIZE 256
#define SHIFT 3
using namespace std;

float* generateSimpleCyclicMatrix(int n, int m) {
	srand(time(0));
	float* a = new float[n * m];
	vector<int> vec;

	for (int i = 1; i * i < MAX_SHIFT * MAX_SHIFT; ++i) {
		if (MAX_SHIFT % i == 0) {
			vec.push_back(i);
		}
	}
	vector<int> vec2;
	for (int i = 0; i < m; ++i) {
		int k = rand() % vec.size();
		vec2.push_back(vec[k]);
		vector<float> current_vector;
		for (int j = 0; j < vec[k]; ++j) {
			current_vector.push_back(rand() % 10);
		}
		for (int j = 0; j < n / vec[k]; ++j) {
			for (int t = 0; t < vec[k]; ++t) {
				a[i * n + j * vec[k] + t] = current_vector[t];
			}
		}
	}
	a[1] = 1;
	return a;
}

float* generateCyclicShiftMatrix(float* a, int n, int m, int shift) {
	float* b = new float[n * m];
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n - shift; ++j) {
			b[i * n + j] = a[shift + i * n + j];
		}
		for (int j = n - shift, k = 0; k < shift && j < n; ++j, ++k) {
			b[i * n + j] = a[i * n + k];
		}
	}
	return b;
}

void findShifts(float* a, float* b, int& host_res, int n, int m) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			host_res &= b[i * n + j] == a[i * n + (j + SHIFT) % n];
		}
	}
}

__global__ void kernel(float* dev_a, float* dev_b, int* dev_res, int n) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int dev_res_private = 1;
	__shared__ int dev_res_shared;
	for (int j = 0; j < n; ++j) {
		dev_res_private &= dev_b[row * n + j] == dev_a[row * n + (j + SHIFT) % n];
	}
	if (threadIdx.x == 0) dev_res_shared = 1;
	__syncthreads();
	//dev_res_shared &= dev_res_private;
	atomicAnd(&dev_res_shared, dev_res_private);
	__syncthreads();
	if (threadIdx.x == 0) {
		//*dev_res &= dev_res_shared;
		atomicAnd(dev_res, dev_res_shared);//(dev_res,dev_res_private)
	}
}

__global__ void kernel_shared(float* dev_a, float* dev_b, int* dev_res, int n) {
	int tx = threadIdx.x, bx = blockIdx.x;
	int row = bx * blockDim.x + tx;
	int dev_res_private = 1;
	__shared__ int dev_res_shared;
	__shared__ float cache_a[256][32], cache_b[256][32];// Maybe better use cache_a[8][32], cache_b[8][32]; OR cache_a[256][16], cache_b[256][16]; 
	for (int k = 0; k < n/32; k++) {
		//заполняем кеши
		for (int p = 0; p < 32; p++ ){
			cache_a[tx/32 + p*8][tx%32] = dev_a[];
			cache_b[tx/32 + p*8][tx%32] = dev_b[];
		}
		for (int j = 0; j < 256; ++j) {
			dev_res_private &= cache_b[bx * blockDim.x][j] == cache_a[bx * blockDim.x][j];
		}
	}
	if (threadIdx.x == 0) dev_res_shared = 1;
	__syncthreads();
	//dev_res_shared &= dev_res_private;
	atomicAnd(&dev_res_shared, dev_res_private);
	__syncthreads();
	if (threadIdx.x == 0) {
		//*dev_res &= dev_res_shared;
		atomicAnd(dev_res, dev_res_shared);//(dev_res,dev_res_private)
	}
}



int main(int argc, char** argv)
{
	int n = 1024, m = 26500, shift = 1;
	float* a = generateSimpleCyclicMatrix(n, m);
	float* b = generateCyclicShiftMatrix(a, n, m, shift);
	int res_CPU = true, *res_GPU = new int[1];
	res_GPU[0] = true;
	cudaEvent_t startCUDA, stopCUDA;
	cudaEventCreate(&startCUDA);
	cudaEventCreate(&stopCUDA);
	clock_t startCPU;
	float elapsedTimeCUDA, elapsedTimeCPU;
	startCPU = clock();
	findShifts(a, b, res_CPU, n, m);
	elapsedTimeCPU = (double)(clock() - startCPU) / CLOCKS_PER_SEC;
	cout << "CPU time = " << elapsedTimeCPU*1000 << " ms\n";

	dim3 gridSize = dim3(m / BLOCK_SIZE, 1, 1);
	dim3 blockSize = dim3(BLOCK_SIZE, 1, 1);
	int* dev_res;
	float* dev_a, * dev_b;
	int sz = n * m * sizeof(float), sz_shifts = sizeof(int);
	cudaMalloc(&dev_a, sz);
	cudaMalloc(&dev_b, sz);
	cudaMalloc(&dev_res, sz_shifts);
	cudaMemcpy(dev_a, a, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_res, res_GPU, sz_shifts, cudaMemcpyHostToDevice);

	cudaEventRecord(startCUDA, 0);
	kernel_shared <<<gridSize, blockSize>>> (dev_a, dev_b, dev_res, n);
	cudaEventRecord(stopCUDA, 0);
	cudaEventSynchronize(stopCUDA);
	cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);
	cout << "CUDA time = " << elapsedTimeCUDA << " ms\n";

	cudaMemcpy(res_GPU, dev_res, sz_shifts, cudaMemcpyDeviceToHost);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_res);
	cout << endl << "CPU result : " << res_CPU  <<  endl;
	cout << endl << "GPU result : " << res_GPU[0] << endl;
	return 0;
}
