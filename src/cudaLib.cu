
#include "cudaLib.cuh"
#include <curand_kernel.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int idx = threadIdx.x + blockIdx.x *blockDim.x;
	// printf("count:%d, %d %d\n",threadIdx.x, blockIdx.x, blockDim.x);
	if(idx<size){
		y[idx] = scale*x[idx]+y[idx];
	}
}



int runGpuSaxpy(int vectorSize) {
	
	std::cout << "Hello GPU Saxpy!\n";


	//	Insert code here

	// declare the host array
	float *host_a, *host_b, *host_c;
	float scale = 2.0f;
	host_a = (float *) malloc(vectorSize * sizeof(float));
	host_b = (float *) malloc(vectorSize * sizeof(float));
	host_c = (float *) malloc(vectorSize * sizeof(float));
	vectorInit(host_a, vectorSize);
	vectorInit(host_b, vectorSize);
	std::memcpy(host_c, host_b, vectorSize * sizeof(float));
	// declare memory array
	float *device_a, *device_b;

	// allocate the device memory
	cudaMalloc((void**)&device_a, vectorSize * sizeof(float));
	cudaMalloc((void**)&device_b, vectorSize * sizeof(float));
	
	// copy the data from host to device
	cudaMemcpy(device_a,host_a,vectorSize * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(device_b,host_b,vectorSize * sizeof(float),cudaMemcpyHostToDevice);

	// operation
	int threadsPerBlock = 256; 
	int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
	saxpy_gpu<<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, scale, vectorSize);

	// copy from device to host
	cudaMemcpy(host_c,device_b,vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

	//verify
	int errorCount = verifyVector(host_a, host_b, host_c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";
	
	// free mem
	cudaFree(device_a);
	cudaFree(device_b);	

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. 
 The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int idx = threadIdx.x + blockIdx.x *blockDim.x;
	if(idx<sampleSize){
		curandState state;
		curand_init(clock64(), idx, 0, &state);
		uint64_t hitCount = 0;

		for (int i=0;i<pSumSize;i++){
			float x = curand_uniform(&state);
			float y = curand_uniform(&state);

			if ( int(x * x + y * y) <= 0 ) {
				hitCount++;
			}
		}

		pSums[idx] = hitCount;
	}
	
	
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int idx = threadIdx.x + blockIdx.x *blockDim.x;
	for(int i=0;i<pSumSize;i++){
		int tmp = i%reduceSize;
		if(tmp==idx){
			totals[tmp]+=pSums[i];
		}
		
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	printf("%d, %d, %d, %d\n",generateThreadCount,sampleSize,reduceThreadCount, reduceSize);
	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();

	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;
	
	// Insert code here
	unsigned long* host_hitCount;
	host_hitCount = (unsigned long*) calloc(generateThreadCount, sizeof(unsigned long));
	unsigned long* device_hitCount;
	int threadsPerBlock = 128; 
	int blocksPerGrid = (generateThreadCount + threadsPerBlock - 1) / threadsPerBlock;

	int pSumSize = sampleSize / generateThreadCount;
	cudaMalloc((void**)&device_hitCount, generateThreadCount * sizeof(unsigned long));
	cudaMemcpy(device_hitCount,host_hitCount,generateThreadCount * sizeof(unsigned long),cudaMemcpyHostToDevice);
	generatePoints<<<blocksPerGrid, threadsPerBlock>>>(device_hitCount,pSumSize,sampleSize);
	cudaDeviceSynchronize();

	unsigned long* host_reduceHitCount;
	host_reduceHitCount = (unsigned long*) calloc(reduceThreadCount, sizeof(unsigned long));

	unsigned long * device_reduceHitCount;
	cudaMalloc((void**)&device_reduceHitCount, reduceThreadCount * sizeof(unsigned long));

	// cudaMemcpy(device_hitCount,host_hitCount,generateThreadCount * sizeof(unsigned long),cudaMemcpyHostToDevice);
	reduceCounts<<<blocksPerGrid, threadsPerBlock>>>(device_hitCount,device_reduceHitCount,generateThreadCount,reduceSize);
	cudaMemcpy(host_reduceHitCount,device_reduceHitCount,reduceThreadCount * sizeof(unsigned long), cudaMemcpyDeviceToHost);

	for(int i=0;i<reduceSize;i++){
		approxPi += host_reduceHitCount[i];
	}

	approxPi = ((double)approxPi / sampleSize);
	approxPi = approxPi * 4.0f;
	// std::cout<<approxPi<<std::endl;
	cudaFree(device_hitCount);	
	return approxPi;
}
