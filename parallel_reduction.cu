/*
 ============================================================================
 Name			: Fundamentals of GPU programming
 Author			: David Celny
 Date			: 31.03.2019
 Description	: paralell reduction (atomic,cascaded,harris)
 Tasks			: For a floating point vector of 40MB size
	-> Write a CPU implementation of the reduction algorithm (sum of all the
		elements). Use it for checking all of GPU outputs!!!
	-> Write a GPU implementation of the reduction algorithm using the atomic add
		functions on one variable in the global memory
	-> Write a GPU implementation of the reduction algorithm using a “cascaded
		algorithm”, use all of the following: sub-reduction on the thread level using
		the registers and ILP, sub-reduction on the block level using the atomic add
		functions in the shared memory, final reduction on the grid level in the global
		memory.
	-> Experiment with the number of blocks and find the fastest
		configuration.
	-> Analyze the results
				: extension for better performing algorithm by Harris
	=> Implement the reduction algorithm on GPU using the cascaded algorithm of
		Harris with the fixed number of blocks and all the optimizations discussed
		today (excluding the warp shuffle optimization). Do it in several separate
		kernel calls in order to be able to process arrays of arbitrary length
	=> Compare its performance with the CPU result and the fastest version of the
		atomic reduction kernel implemented in the previous homework. Analyze the results
 ============================================================================
 */


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime_api.h"

#ifndef RND_WIDTH
#define RND_WIDTH 10
#endif // RND_WIDTH

#ifndef RND_SHIFT
#define RND_SHIFT 5
#endif // RND_SHIFT

#define CUDA_ERROR_CHECK

// the controll of implementation technique
#define ATOMIC 1
#define CASCADED 2
#define HARRIS 3

#ifndef RED_VER
// #define RED_VER ATOMIC // <- modify this to change operation
// #define RED_VER CASCADED // <- modify this to change operation
#define RED_VER HARRIS // <- modify this to change operation
#endif // RED_VER

#ifndef TILE_SIZE
#define TILE_SIZE 512 //  <- modify this to change tiled approach size
#endif // TILE_SIZE

#ifndef ILP
#define ILP 16   // number of the work ech thread calculate
#endif // ILP

/* === Error checking utility section === */
#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaSafeKernell()    __cudaCheckError( __FILE__, __LINE__ )

// constant memory declaration
__constant__ unsigned int d_A_size_x;
__constant__ unsigned int d_A_size_y;

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
void vector_reduce_c (unsigned int A_size_x, float *vect_in, float *reduce_out)
/*
 * the CPU implementation of vector reduction
 */
{
	unsigned int i;
	float tmp_sum = 0.0;
	
	for (i = 0; i < A_size_x; i++)
	{ 
		tmp_sum += vect_in[i];
	}
	reduce_out[0] = tmp_sum;
}

/* = the handling code = */

void get_zero_mat(unsigned int size_x, unsigned int size_y, float *mat)
/*
 * fill the given array with zeros
 * used for the final multiplication matrix initialisation in case of multiple calculation steps
 */
{
	unsigned int i;

	// check the correctness - should be faster
	// mat = memset(0,size_x*size_y*sizeof(float));
	
	for (i = 0; i < size_x*size_y; i++)
	{ // operate on single row		
		mat[i] = 0.0 ;		
	}
	return;
}

void get_rnd_mat(unsigned int size_x, unsigned int size_y, float *mat)
/*
 * fill the given array with random numbers
 *  random generator spread is set with RAND_WIDTH and RAND_SHIFT
 */
{
	unsigned int i,j;

	for (i = 0; i < size_x; i++)
	{ // walk through rows
		for (j = 0; j < size_y; j++)
		{ // walk through collumns
			mat[j + i*size_y] = (float)RND_WIDTH*(rand()/(float)(RAND_MAX)) - RND_SHIFT ;
// 			mat[j + i*size_y] = 1; // DEBUG
		}
	}
	return;
}

void get_rnd_mat_padd(unsigned int size_x, unsigned int size_y, unsigned int size_p_x, unsigned int size_p_y, float *mat)
/*
 * fill the given array with random numbers and padd the the rest <size_p_x size_p_y with zeros
 *  random generator spread is set with RAND_WIDTH and RAND_SHIFT
 */
{
	unsigned int i,j;

	for (i = 0; i < size_p_x; i++)
	{ // walk through rows
		if (i < size_x)
		{
			for (j = 0; j < size_p_y; j++)
			{ // walk through collumns
				if (j < size_y)
				{
// 					mat[j + i*size_p_y] = (float)RND_WIDTH*(rand()/(float)(RAND_MAX)) - RND_SHIFT ;
					mat[j + i*size_p_y] = 1.0 ; // DEBUG
				}
				else
				{
					mat[j + i*size_p_y] = 0.0;
				}
			}
		}
		else
		{
			for (j = 0; j < size_p_y; j++)
			{
				mat[j + i*size_p_y] = 0.0;
			}
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
	for (i = 0; i < size_x; i++)
	{ // walk through rows
		printf("( ");
		for (j = 0; j < size_y; j++)
		{ // walk through colls
			printf("%f ",mat[j + i*size_y]);
		}
		printf(")\n");
	}
	printf("\n");

	return;
}

int check_result(unsigned int size_x, unsigned int size_y, float *mat_host, float*mat_dev, bool output=false)
/*
 * the verification function for result checking
 * 	samples both matrixes no mather how big
 * 			-> the problematic corner values are deffinitely sampled
 * 			-> the middle part is partially sampled (proportionaly to size)
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
	// middle siede indexes
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
			printf("Difference in CPU/GPU comparison at index: %d, x=%d y=%d \n",sample_idx[i],sample_idx[i]/size_x,sample_idx[i]%size_x );
			return -1;
		}
		else if (output)
		{
			printf("at [x=%4d, y=%4d] D=%16.12f H=%16.12f \n",sample_idx[i]/size_x,sample_idx[i]%size_x,mat_dev[sample_idx[i]], mat_host[sample_idx[i]]);
		}
	}
	return 0;
}

int check_result_all(unsigned int size_x, unsigned int size_y, float *mat_host, float*mat_dev, float prec=1e-4, bool output=false)
/*
 * the verification function for result checking
 * 	samples both matrixes no mather how big
 * 			-> the problematic corner values are deffinitely sampled
 * 			-> the middle part is partially sampled (proportionaly to size)
 * 	! beware oversampling for smaller matrixes
 * 	default output is used for printing
 */
{
	unsigned int i,j;
	int tmp_status = 0;
	for (i = 0; i < size_x; i++)
	{ // walk through rows
		for (j = 0; j < size_y; j++)
		{ // walk through colls
			if(abs(mat_host[i*size_y + j] - mat_dev[i*size_y + j])>prec)
			{
				if (output)
				{
					printf("Error in [%d,%d] with prec %f: %f != %f absdiff: %f\n", i, j, prec, mat_host[i*size_y + j], mat_dev[i*size_y + j], abs(mat_host[i*size_y + j] - mat_dev[i*size_y + j]));
				}
				tmp_status++;
			}
		}
	}
	printf("encounterd %d errors \n",tmp_status);
	return tmp_status;
}

int check_result_all_padd(unsigned int size_x, unsigned int size_y, unsigned int size_p_x, unsigned int size_p_y, float *mat_host, float*mat_dev, float prec=1e-4, bool output=false)
/*
 * the verification function for result checking
 * 	samples both matrixes no mather how big
 * 			-> the problematic corner values are deffinitely sampled
 * 			-> the middle part is partially sampled (proportionaly to size)
 * 	! beware oversampling for smaller matrixes
 * 	default output is used for printing
 */
{
	unsigned int i,j;
	int tmp_status = 0;
	for (i = 0; i < size_x; i++)
	{ // walk through rows
		for (j = 0; j < size_y; j++)
		{ // walk through colls
			if(abs(mat_host[i*size_p_y + j] - mat_dev[i*size_p_y + j])>prec)
			{
				if (output)
				{
					printf("Error in [%d,%d] with prec %f: %f != %f absdiff: %f\n", i, j, prec, mat_host[i*size_p_y + j], mat_dev[i*size_p_y + j], abs(mat_host[i*size_p_y + j] - mat_dev[i*size_p_y + j]));
				}
				tmp_status++;
			}
		}
	}
	printf("encounterd %d errors \n",tmp_status);
	return tmp_status;
}

/* == the device code == */

__device__ float subvect_reduce(unsigned int id_i, float *vect_in)
/*
 * the device function for subvector reduction in registers
 * expected the 1D grid  
 */
{	
	float tmp_sum = 0;

#pragma unroll
	for (int i = 0; i < ILP; i++)
	{ 
		tmp_sum += vect_in[id_i+i];
	}
	return tmp_sum;
}

__global__ void vector_cascade_g (float *vect_in, float *reduce_out)
/*
 * The kernel for vector reduction
 * 	 expected to be called 1D grid
 * 	 three step reduction:
 * 						 1) registers reduce per thread - __device__
 * 						 2) shared memory reduce of thread partial results
 * 						 3) reduce of block subresults into global memory
 */
{
	unsigned int id_i =  blockDim.x * blockIdx.x + threadIdx.x; 
	id_i *= ILP; //each thread operates on the ILP number of elements
	 // size check - beware it is crude in this context can still include rubbish that is caught in ILP
	
	__shared__ float block_subres[TILE_SIZE+1];
	
	block_subres[threadIdx.x] = subvect_reduce(id_i, vect_in);
	
// 	printf("%d, %f\n", id_i, block_subres[threadIdx.x]); // DEBUG printout
	atomicAdd(&block_subres[TILE_SIZE], block_subres[threadIdx.x]); // increment each thread to the last element of shared memory
	__syncthreads(); // synch the atomic adds
	
	if (threadIdx.x == 0)
	{
		atomicAdd(reduce_out, block_subres[TILE_SIZE]); // increment the whole block to the global memory
		printf("%d, %f\n", blockIdx.x, block_subres[TILE_SIZE]); // DEBUG printout
	}
	// 
}

__global__ void vector_atomic_g (float *vect_in, float *reduce_out)
/*
 * The kernel for vector reduction
 * 	 expected to be called 1D grid
 * 	 reduce the vecotr into single place in global memory with atomic functions
 */
{
	unsigned int id_i =  blockDim.x * blockIdx.x + threadIdx.x; 
	if (id_i >= d_A_size_y)
	{
		return; // size check		
	}else{
		atomicAdd(reduce_out, vect_in[id_i]); // increment the whole block to the global memory
	}
	
}

template <unsigned int blockSize>
__device__ void subvect_reduce_harris(volatile float *vect_in, unsigned int id_i)
/*
 * the device function for subvector reduction in registers with interwarp unrolling in mind
 * expected the 1D input with < 32 elem
 * Templated with blockSize
 */
{
	if (blockSize >= 64)
	{
		vect_in[id_i] += vect_in[id_i +32];
	}
	if (blockSize >= 32)
	{
		vect_in[id_i] += vect_in[id_i +16];
	}
	if (blockSize >= 16)
	{
		vect_in[id_i] += vect_in[id_i + 8];
	}
	if (blockSize >=  8)
	{
		vect_in[id_i] += vect_in[id_i + 4];
	}
	if (blockSize >=  4)
	{
		vect_in[id_i] += vect_in[id_i + 2];
	}
	if (blockSize >=  2)
	{
		vect_in[id_i] += vect_in[id_i + 1];	
	}
}

template <unsigned int blockSize>
__global__ void vector_harris_g (float *vect_in, float *reduce_out)
/*
 * The kernel for vector reduction
 * 	 expected to be called 1D grid
 * 	 reduce the vecotr into single place in global memory with haris algorithm
 * 	 BEWARE the requeste # of blocks should be halved !!! 
 */
{
	unsigned int id_i =  blockDim.x * blockIdx.x * 2 + threadIdx.x; 
	
	extern __shared__ float block_subres[];
	
	block_subres[threadIdx.x] = vect_in[id_i] + vect_in[id_i+blockDim.x];
	__syncthreads();
	
	// reduction step in subresults
// 	for (int k=block_dim.x/2; k>32; s >>=1) // bit shift = /2
// 	{
// 		if( threadIdx.x < k )
// 		{
// 			block_subres[threadIdx.x] += block_subres[threadIdx.x + k];
// 		}
// 		__syncthreads();
// 	}
	
	if (blockSize >= 512)
	{
		if (threadIdx.x < 256)
		{
			block_subres[threadIdx.x] += block_subres[threadIdx.x + 256];	
			__syncthreads();
		}
	}
	if (blockSize >= 256)
	{
		if (threadIdx.x < 128)
		{
			block_subres[threadIdx.x] += block_subres[threadIdx.x + 128];	
			__syncthreads();
		}
	}
	if (blockSize >= 128)
	{
		if (threadIdx.x < 64)
		{
			block_subres[threadIdx.x] += block_subres[threadIdx.x + 64];	
			__syncthreads();
		}
	}
	if (threadIdx.x < 32)
	{
		subvect_reduce_harris<blockSize>(block_subres, threadIdx.x);
	}
	
	if (threadIdx.x == 0) // save to the individual global memory places -> no atomic required
	{
// 		printf("%d, %f\n", blockIdx.x, block_subres[0]); // DEBUG printout
// 		reduce_out[blockIdx.x] = block_subres[0]; // print to glbal vector variant -> summ with cpu 
		atomicAdd(reduce_out, block_subres[0]); // atomic add to global memory
	}
}

/* == the main call code == */

int main( int argc, char *argv[] )
/*
 * main function executing the kernell call and GPU properties output
 */
{
	const long int seed = 123456789;

	const unsigned int A_size_x = 1;
	const unsigned int A_size_y = 1000000; // it is a vector
	
	const unsigned int B_size_x = 1; // single reduce result
	const unsigned int B_size_y = 1; // it is only a number
		
	const unsigned int A_size_p_x = 1; // no padding in this dimension is needed
#if RED_VER == CASCADED
	const unsigned int A_size_p_y = TILE_SIZE*ILP*(int)(((A_size_y+TILE_SIZE-1)/TILE_SIZE + ILP -1)/ILP);	
#elif RED_VER == HARRIS
	const unsigned int A_size_p_y = TILE_SIZE*2*(int)(((A_size_y+TILE_SIZE-1)/TILE_SIZE + 2 -1)/2);	
#else
	const unsigned int A_size_p_y = TILE_SIZE*(int)((A_size_y + TILE_SIZE -1)/TILE_SIZE);	
#endif
	const unsigned int B_size_p_x = 1; // no padding in this dimension is needed
	const unsigned int B_size_p_y = 1; //TILE_SIZE*(int)((B_size_y + TILE_SIZE -1)/TILE_SIZE);
		
	size_t A_size_n = A_size_p_x*A_size_p_y*sizeof(float);
	size_t B_size_n = B_size_p_x*B_size_p_y*sizeof(float);
	
	
	int available_device; // the device number ! expect only single GPU coprocessor presence to work as intended

	// host variables
	float *matA = NULL;
	float *matB = NULL;
	float *matB_dev;
	// device variables
	float *d_matA;
	float *d_matB;
	
	// timing
	clock_t cpu_start, cpu_stop;
	cudaEvent_t gpu_start, gpu_start_b, gpu_stop, gpu_stop_b;
	float cpu_time, gpu_time;

	/* get and display device infromation section */
	cudaDeviceProp device_prop;
	cudaSafeCall( cudaGetDevice(&available_device)); //get the device count
	cudaSafeCall( cudaGetDeviceProperties(&device_prop, available_device)); // get the last device properties

// 	printf("*** Coprocessor %s information ***\n", device_prop.name);
// 	printf("*** SM: %i, Gmem: %d MB, Smem/B: %d kB, Cmem: %d kB ***\n",(int)device_prop.multiProcessorCount
// 																	  ,(int)device_prop.totalGlobalMem/1024/1024
// 																	  ,(int)device_prop.sharedMemPerBlock/1024
// 																	  ,(int)device_prop.totalConstMem/1024);
	/* initialization section */
	
	srand(seed);
	
	// init the rest: matB, matB_dev
	matA = (float*) malloc(A_size_n);
	matB = (float*) malloc(B_size_n);
	matB_dev = (float*) malloc(B_size_n);

// 	get_rnd_mat (A_size_x, A_size_y, matA);
// 	get_zero_mat (B_size_x, B_size_y, matB);

// 	display_matrix (A_size_x, A_size_y, matA);
// 	display_matrix (B_size_x, B_size_y, matB);

	get_rnd_mat_padd (A_size_x, A_size_y, A_size_p_x, A_size_p_y, matA);
	get_zero_mat (B_size_p_x, B_size_p_y, matB);
	
// 	display_matrix (A_size_p_x, A_size_p_y, matA);
// 	display_matrix (B_size_p_x, B_size_p_y, matB);
	
	// the device init
	cudaSafeCall(cudaMalloc((void**)&d_matA, A_size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matB, B_size_n));
		
	// constant memory move
	cudaSafeCall(cudaMemcpyToSymbol(d_A_size_x, &A_size_p_x, sizeof(unsigned int)));
	cudaSafeCall(cudaMemcpyToSymbol(d_A_size_y, &A_size_p_y, sizeof(unsigned int))); 
	
	// the data move host->device
	cudaSafeCall(cudaMemcpy(d_matA, matA, A_size_n, cudaMemcpyHostToDevice));
	
	
	// timing
	cudaSafeCall(cudaEventCreate(&gpu_start));
	cudaSafeCall(cudaEventCreate(&gpu_stop));
	
	cudaSafeCall(cudaEventCreate(&gpu_start_b));
	cudaSafeCall(cudaEventCreate(&gpu_stop_b));
	
	/* Kernell call section */
#if RED_VER == CASCADED
	dim3 thread_dim(TILE_SIZE);
	dim3 block_dim( ((A_size_p_y+TILE_SIZE-1)/TILE_SIZE + ILP -1)/ILP );
#elif RED_VER == ATOMIC
	dim3 thread_dim(TILE_SIZE);
	dim3 block_dim( (A_size_p_y+TILE_SIZE-1)/TILE_SIZE );
#else // HARRIS
	dim3 thread_dim(TILE_SIZE);
	dim3 block_dim( (A_size_p_y+TILE_SIZE-1)/TILE_SIZE/2 );
	size_t smem = TILE_SIZE*sizeof(float);
#endif

	/* the reduce kernel */
	cudaSafeCall(cudaEventRecord(gpu_start, 0));
	
#if RED_VER == ATOMIC
	vector_atomic_g<<<block_dim,thread_dim>>>(d_matA, d_matB);
#elif RED_VER == CASCADED
	vector_cascade_g<<<block_dim,thread_dim>>>(d_matA, d_matB);
#elif RED_VER == HARRIS
	switch (TILE_SIZE)
	{
		case 512:
			vector_harris_g<512><<<block_dim,thread_dim,smem>>>(d_matA, d_matB);
			break;
		case 256:
			vector_harris_g<256><<<block_dim,thread_dim,smem>>>(d_matA, d_matB);
			break;
		case 128:
			vector_harris_g<128><<<block_dim,thread_dim,smem>>>(d_matA, d_matB);
			break;
		case 64:
			vector_harris_g<64><<<block_dim,thread_dim,smem>>>(d_matA, d_matB);
			break;
		case 32:
			vector_harris_g<32><<<block_dim,thread_dim,smem>>>(d_matA, d_matB);
			break;
		case 16:
			vector_harris_g<16><<<block_dim,thread_dim,smem>>>(d_matA, d_matB);
			break;
		case 8:
			vector_harris_g<8><<<block_dim,thread_dim,smem>>>(d_matA, d_matB);
			break;
		case 4:
			vector_harris_g<4><<<block_dim,thread_dim,smem>>>(d_matA, d_matB);
			break;
		case 2:
			vector_harris_g<2><<<block_dim,thread_dim,smem>>>(d_matA, d_matB);
			break;
		case 1:
			vector_harris_g<1><<<block_dim,thread_dim,smem>>>(d_matA, d_matB);
			break;
	}
#endif

	cudaDeviceSynchronize();
	cudaSafeCall(cudaEventRecord(gpu_stop, 0));
	cudaSafeCall(cudaEventSynchronize(gpu_stop));
//	cudaSafeKernell(); // not to be included in speed measurement

	cudaSafeCall(cudaEventElapsedTime(&gpu_time,gpu_start,gpu_stop));

	// the data move device->host
	cudaSafeCall(cudaMemcpy(matB_dev, d_matB, B_size_n, cudaMemcpyDeviceToHost));

	// event cleaup
	cudaSafeCall(cudaEventDestroy(gpu_start));
	cudaSafeCall(cudaEventDestroy(gpu_stop));
	
	/* CPU section */
	cpu_start = clock();

	vector_reduce_c(A_size_p_y, matA, matB);

	cpu_stop = clock();

	cpu_time = (cpu_stop-cpu_start)/((float)CLOCKS_PER_SEC);
	/* execution statistics section */
// 	printf("*** level of parallelization ***\n");

	printf("*** matrix: %d,%d ***\n", A_size_x, A_size_y);
	printf("*** padded: %d,%d ***\n", A_size_p_x, A_size_p_y);
	printf("*** block: %d ***\n", block_dim.x);
	printf("*** thread: %d ***\n", thread_dim.x);
	printf("*** ILP: %d ***\n", ILP);
	
	printf("*** time measurement ***\n");
	printf("*** CPU: %f ms\n",cpu_time*1000);
#if RED_VER == ATOMIC
	printf("*** GPU atomic: %f ms\n",gpu_time);	
#elif RED_VER == CASCADED
	printf("*** GPU cached: %f ms\n",gpu_time);	
#elif RED_VER == HARRIS
	printf("*** GPU Harris: %f ms\n",gpu_time);	
#else
	printf(" !!! Unsuported version option: %d !!! \n",RED_VER);
#endif
	printf("*** speedup: %f \n",cpu_time*1000/gpu_time);
	printf("*** bandwidth: %f \n",2*A_size_n/gpu_time/1024/1024/1024); // with 3* '/1024' in GB

// 	printf("%16.12f , %16.12f, %16.12f \n",cpu_time*1000,gpu_time, cpu_time*1000/gpu_time); // for batch execution
	
	/* controll section */

	// display_matrix(B_size_x, B_size_y, matB);
	// display_matrix(B_size_x, B_size_y, matB_dev);
// 	int result= check_result(B_size_x, B_size_y, matB, matB_dev, false); // sampling
	int result= check_result_all(B_size_x, B_size_y, matB, matB_dev, 1e-1, true); // full check
	
	/* celanup section */
	free(matA);
	free(matB);
	free(matB_dev);
	cudaFree(d_matA);
	cudaFree(d_matB);

	return result;
}
