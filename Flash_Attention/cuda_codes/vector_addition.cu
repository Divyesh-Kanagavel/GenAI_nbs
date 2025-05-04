#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

typedef int EL_TYPE;

#define CUDA_CHECK(err) do {cuda_check((err), __FILE__, __LINE__);}while(false)

void cuda_check(cudaError_t error_code, const char* file, int line)
{
    if (error_code!=cudaSuccess)
    {
        fprintf(stderr, "CUDA ERROR %d : %s. In file %s on line %d\n", error_code, 
        cudaGetErrorString(error_code),file,line);
        fflush(stderr);
        exit(error_code);
    }
}

/* cuda code to add two vectors */
__global__ void cuda_vector_add(EL_TYPE* out, EL_TYPE* A, EL_TYPE* B, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        out[i] = A[i] + B[i];
    }
}

void test_vector_add(int N, int block_size)
{
    EL_TYPE *A, *B, *out;
    EL_TYPE *d_A, *d_B, *d_out;

    /* allocate memory for the buffers */
    A = (EL_TYPE*)malloc(sizeof(EL_TYPE) * N);
    B = (EL_TYPE*)malloc(sizeof(EL_TYPE) * N);
    out = (EL_TYPE*)malloc(sizeof(EL_TYPE)*N);
    /* assign random values to A and B */
    for(int i=0;i<N;i++)
    {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }
    /* allocate device memory */
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeof(EL_TYPE)*N));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeof(EL_TYPE)*N));
    CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(EL_TYPE)*N));

    /* transfer arrays to device */
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(EL_TYPE)*N,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(EL_TYPE)*N,cudaMemcpyHostToDevice));
    /* define kernel block, grid parameters */
    int num_blocks = ceil((float)N / block_size);
    printf("Vectod add - N  : %d will be processed by %d blocks of size %d\n", N,
    num_blocks, block_size);

    dim3 grid(num_blocks, 1, 1);
    dim3 block(block_size, 1, 1);

    /* event recorder variables to measure cuda runtime */
    cudaEvent_t startKernel, endKernel;
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&endKernel));

    CUDA_CHECK(cudaEventRecord(startKernel));
    cuda_vector_add<<<grid, block>>> (d_out, d_A,d_B, N); /* call cuda addition function with N threads */
    CUDA_CHECK(cudaEventRecord(endKernel));

    // Check for launch errors
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventSynchronize(endKernel));

    float milliseconds_kernel = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, startKernel, endKernel));
    printf("Cuda time elapsed for vector addition  = %f ms.\n", milliseconds_kernel);
    CUDA_CHECK(cudaMemcpy(out,d_out, sizeof(EL_TYPE) * N, cudaMemcpyDeviceToHost));
    /* free memory in the device */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_out));

    /* verify the result! */

    struct timeval start_check, end_check;
    gettimeofday(&start_check, NULL);

    for (int i = 0; i < N; i++)
    {
        // Check if the result is correct
        if (out[i] != A[i] + B[i])
        {
            printf("Error at index %d: %d != %d + %d\n", i, out[i], A[i], B[i]);
            exit(1);
        }
    }
    // Calculate elapsed time
    gettimeofday(&end_check, NULL);
    float elapsed = (end_check.tv_sec - start_check.tv_sec) * 1000.0 + (end_check.tv_usec - start_check.tv_usec) / 1000.0;
    printf("Vector Add - Check elapsed time: %f ms\n", elapsed);
    printf("Vector add result - OK!\n");

    /* free the memory on the host */

    free(A);
    free(B);
    free(out);
}

int main()
{
    srand(0); /*set the seed */
    test_vector_add(1000000, 128);
    return 0;
}

