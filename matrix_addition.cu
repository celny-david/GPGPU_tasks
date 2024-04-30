/*
 ============================================================================
 Name			: Fundamentals of GPU programming
 Author			: David Celny
 Date			: 26.10.2018
 Description	: arbitrary size matrix addition with comparison and control
 Tasks			: Implement parallel version of the matrix addition code on GPU
for matrices of arbitrary size.
				: Compare execution time (excluding the data transfers) for
matrices of 10x10, 100x100, 1000x1000, 500x2000,100x10000 float random elements
				: Compare analogous execution times for a CPU version of matrix addition
				: Compare execution time for three different execution
configurations (16 x 16, 16 x 32, 32 x 16) for the 100 x 10000 case on GPU
 ============================================================================
 */

#include <stddef.h>
#include <stdio.h>
#include "cuda_runtime_api.h"

#ifndef RND_WIDTH
#define RND_WIDTH 10
#endif // RND_WIDTH

#ifndef RND_SHIFT
#define RND_SHIFT 5
#endif // RND_SHIFT

#define CUDA_ERROR_CHECK

/* === Error checking utility section === */
#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaSafeKernell()    __cudaCheckError( __FILE__, __LINE__ )

// constant memory declaration
__constant__ unsigned int d_size_x;
__constant__ unsigned int d_size_y;

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "CUDA ERROR %i at %s: %i of type: %s\n",
                 err, file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "CUDA ERROR %i at %s: %i of type: %s\n",
        		err, file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "CUDA ERROR %i with sync at %s: %i of type: %s\n",
                 err, file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

/* === The program section === */
/* == the host code == */
__host__ __device__ void element_add(unsigned int ind,float *matA, float *matB, float *matC)
/*
 * the device and host function for elementwise addition into matrixes
 */
{
	matC[ind] = matA[ind] + matB[ind]; // the very addition
}

void matrix_add_c (unsigned int size_x, unsigned int size_y,float *matA, float *matB, float *matC)
/*
 * the CPU implementation of matrix addition
 * the matrixes are in flattened form -> index as vectors
 *   								  -> beware row/coll order
 */
{
	unsigned int i,j;

	for (i = 0; i < size_y; i++)
	{ // operate on single row
		for (j = 0; j < size_x; j++)
		{ // walk through rows
			element_add(j + i*size_x, matA,matB,matC);
		}
	}
}

/* = the handling code = */
void get_rnd_mat(unsigned int size_x, unsigned int size_y, float *mat)
/*
 * fill the given array with random numbers
 *  random generator spread is set with RAND_WIDTH and RAND_SHIFT
 */
{
	unsigned int i,j;

	for (i = 0; i < size_y; i++)
	{ // operate on single row
		for (j = 0; j < size_x; j++)
		{ // walk through rows
			mat[j + i*size_x] = RND_WIDTH*(rand()/(float)(RAND_MAX)) - RND_SHIFT ;
		}
	}
	return;
}
void display_matrix(unsigned int size_x, unsigned int size_y, float *mat)
/*
 * utility for displaying small matrixes
 */
{
	unsigned int i,j;
	for (i = 0; i < size_y; i++)
	{ // operate on single row
		printf("( ");
		for (j = 0; j < size_x; j++)
		{ // walk through rows
			printf("%f ",mat[j + i*size_x]);
		}
		printf(")\n");
	}
	printf("\n");
	return;
}
int check_result(unsigned int size_x, unsigned int size_y, float *mat_host, float*mat_dev, bool output=false)
/*
 * the verification function for result checking
 * 	samples both matrixes no matter how big
 * 			-> the problematic corner values are definitely sampled
 * 			-> the middle part is partially sampled (proportionally to size)
 * 	! beware oversampling for smaller matrixes
 * 	default output is used for printing
 */
{
	unsigned int i, tmp;
	const unsigned int idx_cnt = 8+log2(1.0*size_x*size_y); //sampling quantity
	unsigned int sample_idx[idx_cnt]; // the sampled indexes

	// corner indexes
	sample_idx[0] = 0;
	sample_idx[1] = size_x - 1;
	sample_idx[2] = (size_y - 1)*size_x;
	sample_idx[3] = size_x*size_y - 1;
	// middle side indexes
	sample_idx[4] = size_x/2;
	sample_idx[5] = size_y/2 *size_x;
	sample_idx[6] = size_y/2*(size_x+1);
	sample_idx[7] = size_x*size_y - size_x/2;
	// the sampled middle of matrix
	if (size_x>2 && size_y>2)
	{
		for (i = 8; i < idx_cnt; i++)
		{
			tmp = (int)((size_x)*(rand()/(float)(RAND_MAX)));			// random x index of size_x
			tmp += (int)((size_y)*(rand()/(float)(RAND_MAX)))*size_x; // and random y index of size_y
			sample_idx[i] = tmp;
		}
	}
	printf(" *** Sampling %d elements from matrix: ***\n     4 corners, 4 middle sides, %d insides\n", idx_cnt, idx_cnt-8);
	for (i = 0; i < idx_cnt; i++)
	{
		tmp = mat_dev[sample_idx[i]] - mat_host[sample_idx[i]]; // perform subtraction
		if (tmp!= 0) // beware the epsilon differences - not solved
		{// SELFNOTE for (-5,5) the situation seems fine
			printf("Difference in CPU/GPU comparison at index: %d, x=%d y=%d \n",sample_idx[i],sample_idx[i]%size_x,sample_idx[i]/size_x );
			return -1;
		}
		else if (output)
		{
			printf("at [x=%4d, y=%4d] D=%16.12f H=%16.12f \n",sample_idx[i]%size_x,sample_idx[i]/size_x,mat_dev[sample_idx[i]], mat_host[sample_idx[i]]);
		}
	}
	return 0;
}


/* == the device code == */
__global__ void matrix_add_g (float *matA, float *matB, float *matC)
/*
 * The kernel for addition of two matrixes
 * ! elementwise addition with size input size_x, size_y
 * 	 expected to be called as two dimensional grid
 *   the matrixes are in flattened form -> index as vectors
 *   									-> beware row/coll order
 */
{
	unsigned int idx =  blockDim.x * blockIdx.x + threadIdx.x; // x_dim index
	if(idx> d_size_x) return;
	idx += (blockDim.y * blockIdx.y + threadIdx.y)*d_size_x; // y_dim index increment
	if(idx> d_size_x*d_size_y) return;

	element_add(idx, matA,matB,matC);

}

int main()
/*
 * main function executing the kernel call and GPU properties output
 */
{
	const unsigned int size_x = 100;
	const unsigned int size_y = 10000;
	size_t size_n = size_x*size_y*sizeof(float);
	const long int seed = 123456789;

	unsigned int thread_count_x = 32; // number of threads used per block in x
	unsigned int thread_count_y = 32; // number of threads used per block in y

	int awailable_device; // the device number ! expect only single GPU coprocessor presence to work as intended

	// host variables
	float *matA = NULL;
	float *matB = NULL;
	float *matC = NULL;
	float *matC_dev;
	// device variables
	float *d_matA;
	float *d_matB;
	float *d_matC;

	// timing
	clock_t cpu_start, cpu_stop;
	cudaEvent_t gpu_start, gpu_stop;
	float cpu_time, gpu_time;

	/* get and display device information section */
	cudaDeviceProp device_prop;
	cudaSafeCall( cudaGetDevice(&awailable_device)); //get the device count
	cudaSafeCall( cudaGetDeviceProperties(&device_prop, awailable_device)); // get the last device properties

	printf("*** Coprocessor %s information ***\n", device_prop.name);
	printf("*** SM: %i, Gmem: %d MB, Smem/B: %d kB, Cmem: %d kB ***\n",(int)device_prop.multiProcessorCount
																	  ,(int)device_prop.totalGlobalMem/1024/1024
																	  ,(int)device_prop.sharedMemPerBlock/1024
																	  ,(int)device_prop.totalConstMem/1024);
	/* Initialisation section */
	srand(seed);
	matA = (float*) malloc(size_n);
	matB = (float*) malloc(size_n);
	matC = (float*) malloc(size_n);
	matC_dev = (float*) malloc(size_n);
	get_rnd_mat(size_x,size_y,matA);
//	display_matrix(size_x, size_y, matA);
	get_rnd_mat(size_x,size_y,matB);
//	display_matrix(size_x, size_y, matB);

	// the device init
	cudaSafeCall(cudaMalloc((void**)&d_matA, size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matB, size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matC, size_n));
	// constant memory move
	cudaSafeCall(cudaMemcpyToSymbol(d_size_x, &size_x, sizeof(unsigned int)));
	cudaSafeCall(cudaMemcpyToSymbol(d_size_y, &size_y, sizeof(unsigned int)));

	// the data move host->device
	cudaSafeCall(cudaMemcpy(d_matA, matA, size_n, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(d_matB, matB, size_n, cudaMemcpyHostToDevice));
	// timing
	cudaSafeCall(cudaEventCreate(&gpu_start));
	cudaSafeCall(cudaEventCreate(&gpu_stop));

	/* Kernel call section */
	dim3 thread_dim(thread_count_x,thread_count_y);
	dim3 block_dim(ceil((float)size_x/thread_count_x),ceil((float)size_y/thread_count_y));

	cudaSafeCall(cudaEventRecord(gpu_start, 0));

	matrix_add_g<<<block_dim,thread_dim>>>(d_matA, d_matB, d_matC);
	cudaDeviceSynchronize();
	cudaSafeCall(cudaEventRecord(gpu_stop, 0));
	cudaSafeCall(cudaEventSynchronize(gpu_stop));
//	cudaSafeKernell(); // not to be included in speed measurement

	cudaSafeCall(cudaEventElapsedTime(&gpu_time,gpu_start,gpu_stop));
	// the data move device->host
	cudaSafeCall(cudaMemcpy(matC_dev, d_matC, size_n, cudaMemcpyDeviceToHost));

	// event cleaup
	cudaSafeCall(cudaEventDestroy(gpu_start));
	cudaSafeCall(cudaEventDestroy(gpu_stop));
	/* CPU section */
	cpu_start = clock();
	matrix_add_c(size_x, size_y, matA, matB, matC);
	cpu_stop = clock();

	cpu_time = (cpu_stop-cpu_start)/((float)CLOCKS_PER_SEC);
	/* execution statistics section */
	printf("*** level of parallelization ***\n");
	printf("*** matrix: %d,%d ***\n", size_x, size_y);
	printf("*** block: %d,%d ***\n", block_dim.x, block_dim.y);
	printf("*** thread: %d,%d ***\n", thread_dim.x, thread_dim.y);
	printf("*** time measurement ***\n");
	printf("*** CPU: %f ms\n",cpu_time*1000);
	printf("*** GPU: %f ms\n",gpu_time);
	printf("*** speedup: %f \n",cpu_time*1000/gpu_time);

	/* control section */
	// display_matrix(size_x, size_y, matC);
	// display_matrix(size_x, size_y, matC_dev);
	int result= check_result(size_x, size_y, matC, matC_dev, true);

	/* Cleanup section */
	free(matA);
	free(matB);
	free(matC);
	free(matC_dev);
	cudaFree(d_matA);
	cudaFree(d_matB);
	cudaFree(d_matC);

	return result;
}
