/*
 ============================================================================
 Name			: Fundamentals of GPU programming
 Author			: David Celny
 Date			: 31.12.2018
 Description	: tiled matrix multiplication
 Tasks			: Implement the local tiled matrix multiplication algorithm for M and N
matrices (M having size 3000x1000 and N having size1000x5000) with the
shared memory usage on device for matrices of arbitrary size initialized with
random floating point numbers
				: Compare execution time for versions with/without shared memory
				: Compare execution time for the host and device implementations
				: Show dependence of the execution time on TILE_WIDTH. Determine
parameters yielding the shortest execution time. Study the dependence of the
speedup-factor of the GPU vs CPU execution on the matrix size.
 Bonus Tasks	: Implement a variant where the input matrices are put into texture memory.
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
#define NAIVE 0
#define SHARED 1
#define CACHED 2

#ifndef MULL_VER
#define MULL_VER SHARED // <- modify this to change operation
#endif // MULL_VER

// TILE_SIZE is only valid for tiled
#ifndef TILE_SIZE
#define TILE_SIZE 16 //  <- modify this to change tiled approach size max is 32 which corresponds to 32*32 blocks
#endif //TILE_SIZE

#ifndef TIME_REP
#define TIME_REP 100 //  <- modify to change the number of repetition
#endif // TIME_REP

/* === Error checking utility section === */
#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaSafeKernell()    __cudaCheckError( __FILE__, __LINE__ )

// constant memory declaration
__constant__ unsigned int d_A_size_x;
__constant__ unsigned int d_A_size_y;
__constant__ unsigned int d_B_size_x;
__constant__ unsigned int d_B_size_y;


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
inline void element_mat_mat_mul_c(unsigned int id_i, unsigned int id_j, unsigned int id_k, float *matA, float *matB, float *matC, unsigned int A_size_y, unsigned int B_size_y)
/*
 * the CPU function for elemnt matrix-matrix multiplication
 * with the non-square sized matrices the sizes of final A*B is [A_size_x,B_size_y]
 * operates on the nontransposed second matrix (has to iterate through columns of B)
 */
{
	matC[id_i*B_size_y + id_j] += matA[A_size_y*id_i + id_k] * matB[id_j + B_size_y*id_k];
}


void matrix_matrix_mul_c (unsigned int A_size_x, unsigned int A_size_y, unsigned int B_size_y, float *matA, float *matB, float *matC)
/*
 * the CPU implementation of matrix matrix multiplication
 * the matrixes are in flattened form -> index as vectors
 *   								  -> beware row/coll order
 */
{
	unsigned int i,j,k;

	for (i = 0; i < A_size_x; i++)
	{ // walk through rows of A matrix
		for (j = 0; j < B_size_y; j++)
		{ // walk through collumns of B matrix
			for (k = 0; k < A_size_y; k++)
			{// iterate collumns of the A matrix and also colls of B matrix 
				element_mat_mat_mul_c(i,j,k,matA,matB,matC,A_size_y,B_size_y);
			}
		}
	}
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
			mat[j + i*size_y] = RND_WIDTH*(rand()/(float)(RAND_MAX)) - RND_SHIFT ;
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
					mat[j + i*size_p_y] = RND_WIDTH*(rand()/(float)(RAND_MAX)) - RND_SHIFT ;
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

__device__ void submat_mult_g_naive(unsigned int id_i,unsigned int id_j,float *matA, float *matB, float *matC)
/*
 * the device function for result elemnet matrix-matrix multiplication
 * with the non-square sized matrices the sizes of final A*B is [A_size_x,B_size_y]
 * general version with most uncoalesced( esp. matB) memory accesses 
 */
{
	float tmp_res = 0.0;
	
	//#pragma unroll 16
	for(int id_k=0; id_k < d_A_size_y; id_k++)
	{
		tmp_res += matA[id_i*d_A_size_y + id_k] * matB[id_k*d_B_size_y + id_j];
	}
	matC[id_i*d_B_size_y + id_j] = tmp_res;
}

__device__ void submat_mult_g_shared(unsigned int id_i, unsigned int id_j, float *tmp_A_submat, float *tmp_B_submat, float *matA, float *matB, float *matC)
/*
 * the device function for result subelemnet matrix-matrix multiplication
 * with the non-square sized matrices the sizes of final A*B is [A_size_x,B_size_y]
 * temporary result accumulation
 */
{		
	unsigned int id_l, id_k;
	float tmp_res = 0.0;
	
	//#pragma unroll 16
	for(id_l=0; id_l < (d_A_size_y+TILE_SIZE-1)/TILE_SIZE; id_l++) // the grid dim in square TILE_SIZE case is exatly the number of subregions
	{
		// !BEWARE changed perception of indexation and array definition
		tmp_A_submat[threadIdx.x*TILE_SIZE + threadIdx.y] = matA[id_i*d_A_size_y + id_l*TILE_SIZE + threadIdx.y];
		tmp_B_submat[threadIdx.x*TILE_SIZE + threadIdx.y] = matB[(id_l*TILE_SIZE + threadIdx.x)*d_B_size_y + id_j];
		__syncthreads(); // to synch the loading into shared memory

		for(id_k=0; id_k < TILE_SIZE; id_k++)
		{
			tmp_res += tmp_A_submat[threadIdx.x*TILE_SIZE + id_k] * tmp_B_submat[id_k*TILE_SIZE + threadIdx.y];
		}
		__syncthreads(); // to synch the and not calculate with changed data
	}
	matC[id_i*d_B_size_y + id_j] = tmp_res;
}

#if MULL_VER == CACHED // the cached patch

// BEWARE this is a problem for nvcc 12.3 in combination with gcc 13.2
// Mistakenly throws error: texture is not a template
texture<float, cudaTextureType1D, cudaReadModeElementType> texRefA;
texture<float, cudaTextureType1D, cudaReadModeElementType> texRefB;

__device__ void submat_mult_g_cached(unsigned int id_i, unsigned int id_j, float *tmp_A_submat, float *tmp_B_submat, float *matC)
/*
 * the device function for result subelemnet matrix-matrix multiplication
 * with the non-square sized matrices the sizes of final A*B is [A_size_x,B_size_y]
 * temporary result accumulation
 */
{		
	unsigned int id_l, id_k;
	float tmp_res = 0.0;
	
	//#pragma unroll 16
	for(id_l=0; id_l < (d_A_size_y+TILE_SIZE-1)/TILE_SIZE; id_l++) // the grid dim in square TILE_SIZE case is exatly the number of subregions
	{
		// !BEWARE changed perception of indexation and array definition
		tmp_A_submat[threadIdx.x*TILE_SIZE + threadIdx.y] = tex1Dfetch(texRefA,id_i*d_A_size_y + id_l*TILE_SIZE + threadIdx.y);
		tmp_B_submat[threadIdx.x*TILE_SIZE + threadIdx.y] = tex1Dfetch(texRefB,(id_l*TILE_SIZE + threadIdx.x)*d_B_size_y + id_j);
		__syncthreads(); // to synch the loading into shared memory

		for(id_k=0; id_k < TILE_SIZE; id_k++)
		{
			tmp_res += tmp_A_submat[threadIdx.x*TILE_SIZE + id_k] * tmp_B_submat[id_k*TILE_SIZE + threadIdx.y];
		}
		__syncthreads(); // to synch the and not calculate with changed data
	}
	matC[id_i*d_B_size_y + id_j] = tmp_res;
}

__global__ void matrix_matrix_mul_g (float *matC)
/*
 * The kernel for matrix matrix multiplication
 * 	 expected to be called as two dimensional grid
 *   the matrixes are in flattened form -> indexed as vectors
 *   									-> beware row/coll order
 */
{
	__shared__ float tmp_A_submat[TILE_SIZE*TILE_SIZE];
	__shared__ float tmp_B_submat[TILE_SIZE*TILE_SIZE];
	
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
	
	submat_mult_g_cached(id_x, id_y, tmp_A_submat, tmp_B_submat, matC);

}	
#else // normal assignment 04 behaviour

__global__ void matrix_matrix_mul_g (float *matA, float *matB, float *matC)
/*
 * The kernel for matrix matrix multiplication
 * 	 expected to be called as two dimensional grid
 *   the matrixes are in flattened form -> indexed as vectors
 *   									-> beware row/coll order
 */
{
#if MULL_VER == NAIVE
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
#elif MULL_VER == SHARED
	__shared__ float tmp_A_submat[TILE_SIZE*TILE_SIZE];
	__shared__ float tmp_B_submat[TILE_SIZE*TILE_SIZE];
	
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y;  
#else
#endif

#if MULL_VER == NAIVE
	
// 	if(id_x> d_A_size_x) return;
// 	if(id_y> d_B_size_y) return;
	
	submat_mult_g_naive (id_x, id_y, matA, matB, matC);
#elif MULL_VER == SHARED
	submat_mult_g_shared(id_x, id_y, tmp_A_submat, tmp_B_submat, matA, matB, matC);
#else
#endif
}

#endif

int main( int argc, char *argv[] )
/*
 * main function executing the kernell call and GPU properties output
 */
{
	const long int seed = 123456789;

	const unsigned int A_size_x = 1000;
	const unsigned int A_size_y = 1000;
	
	const unsigned int B_size_x = A_size_y;
	const unsigned int B_size_y = 1000;
	
	const unsigned int A_size_p_x = TILE_SIZE*(int)((A_size_x + TILE_SIZE -1)/TILE_SIZE);
	const unsigned int A_size_p_y = TILE_SIZE*(int)((A_size_y + TILE_SIZE -1)/TILE_SIZE);
	
	const unsigned int B_size_p_x = TILE_SIZE*(int)((B_size_x + TILE_SIZE -1)/TILE_SIZE);
	const unsigned int B_size_p_y = TILE_SIZE*(int)((B_size_y + TILE_SIZE -1)/TILE_SIZE);
	
	size_t A_size_n = A_size_p_x*A_size_p_y*sizeof(float);
	size_t B_size_n = B_size_p_x*B_size_p_y*sizeof(float);
	size_t C_size_n = A_size_p_x*B_size_p_y*sizeof(float);
	
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

	/* get and display device infromation section */
	cudaDeviceProp device_prop;
	cudaSafeCall( cudaGetDevice(&awailable_device)); //get the device count
	cudaSafeCall( cudaGetDeviceProperties(&device_prop, awailable_device)); // get the last device properties

// 	printf("*** Coprocessor %s information ***\n", device_prop.name);
// 	printf("*** SM: %i, Gmem: %d MB, Smem/B: %d kB, Cmem: %d kB ***\n",(int)device_prop.multiProcessorCount
// 																	  ,(int)device_prop.totalGlobalMem/1024/1024
// 																	  ,(int)device_prop.sharedMemPerBlock/1024
// 																	  ,(int)device_prop.totalConstMem/1024);
	/* initialization section */
	srand(seed);
	matA = (float*) malloc(A_size_n);
	matB = (float*) malloc(B_size_n);
	matC = (float*) malloc(C_size_n);
	matC_dev = (float*) malloc(C_size_n);

// 	get_rnd_mat (A_size_x, A_size_y, matA);
// 	get_rnd_mat (B_size_x, B_size_y, matB);
// 	get_zero_mat (A_size_x, B_size_y, matC);

// 	display_matrix (A_size_x, A_size_y, matA);
// 	display_matrix (B_size_x, B_size_y, matB);
// 	display_matrix (A_size_x, B_size_y, matC);
	
	get_rnd_mat_padd (A_size_x, A_size_y, A_size_p_x, A_size_p_y, matA);
	get_rnd_mat_padd (B_size_x, B_size_y, B_size_p_x, B_size_p_y, matB);
	get_zero_mat     (A_size_p_x, B_size_p_y, matC);	

// 	display_matrix (A_size_p_x, A_size_p_y, matA);
// 	display_matrix (B_size_p_x, B_size_p_y, matB);
// 	display_matrix (A_size_p_x, B_size_p_y, matC);

	// the device init
	cudaSafeCall(cudaMalloc((void**)&d_matA, A_size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matB, B_size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matC, C_size_n));
	// constant memory move
	cudaSafeCall(cudaMemcpyToSymbol(d_A_size_x, &A_size_p_x, sizeof(unsigned int)));
	cudaSafeCall(cudaMemcpyToSymbol(d_A_size_y, &A_size_p_y, sizeof(unsigned int)));
	cudaSafeCall(cudaMemcpyToSymbol(d_B_size_x, &B_size_p_x, sizeof(unsigned int)));
	cudaSafeCall(cudaMemcpyToSymbol(d_B_size_y, &B_size_p_y, sizeof(unsigned int)));

	
	// timing
	cudaSafeCall(cudaEventCreate(&gpu_start));
	cudaSafeCall(cudaEventCreate(&gpu_stop));

	/* Kernell call section */
	dim3 thread_dim(TILE_SIZE,TILE_SIZE);
	dim3 block_dim(A_size_p_x/TILE_SIZE,B_size_p_y/TILE_SIZE);
	
	cudaSafeCall(cudaEventRecord(gpu_start, 0));
	
#if MULL_VER == CACHED
	cudaArray* cuArrayA;
	cudaArray* cuArrayB;
	cudaMalloc(&cuArrayA, A_size_n);
	cudaMalloc(&cuArrayB, B_size_n);

	cudaMemcpy(cuArrayA, matA, A_size_n, cudaMemcpyHostToDevice);
	cudaMemcpy(cuArrayB, matB, B_size_n, cudaMemcpyHostToDevice);
	
	texRefA.addressMode[0]	= cudaAddressModeClamp;
	texRefA.addressMode[1]	= cudaAddressModeClamp;
	texRefA.filterMode		= cudaFilterModePoint;
	texRefA.normalized		= false;
	
	texRefB.addressMode[0]	= cudaAddressModeClamp;
	texRefB.addressMode[1]	= cudaAddressModeClamp;
	texRefB.filterMode		= cudaFilterModePoint;
	texRefB.normalized		= false;
	
	size_t offsetA = 0;
	size_t offsetB = 0;
	
	cudaBindTexture(&offsetA, texRefA, cuArrayA, A_size_n);	
	cudaBindTexture(&offsetB, texRefB, cuArrayB, B_size_n);	
#else

	// the data move host->device
	cudaSafeCall(cudaMemcpy(d_matA, matA, A_size_n, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(d_matB, matB, B_size_n, cudaMemcpyHostToDevice));
	
#endif
	for (int ii=0; ii<TIME_REP; ii++)
	{
#if MULL_VER == NAIVE || MULL_VER == SHARED 
	matrix_matrix_mul_g<<<block_dim,thread_dim>>>(d_matA, d_matB, d_matC);
// 	cudaSafeKernell();
#elif MULL_VER == CACHED
	matrix_matrix_mul_g<<<block_dim,thread_dim>>>(d_matC);
 	cudaSafeKernell();
#else
	printf(" !!! Unsuported version option: %d !!! \n",MULL_VER);
#endif
	}
	cudaDeviceSynchronize();
	cudaSafeCall(cudaEventRecord(gpu_stop, 0));
	cudaSafeCall(cudaEventSynchronize(gpu_stop));
//	cudaSafeKernell(); // not to be included in speed measurement

	cudaSafeCall(cudaEventElapsedTime(&gpu_time,gpu_start,gpu_stop));

	// the data move device->host
	cudaSafeCall(cudaMemcpy(matC_dev, d_matC, C_size_n, cudaMemcpyDeviceToHost));

	// event cleaup
	cudaSafeCall(cudaEventDestroy(gpu_start));
	cudaSafeCall(cudaEventDestroy(gpu_stop));
	cudaDeviceSynchronize();
	/* CPU section */
	cpu_start = clock();

	matrix_matrix_mul_c(A_size_p_x, A_size_p_y, B_size_p_y, matA, matB, matC);

	cpu_stop = clock();

	cpu_time = (cpu_stop-cpu_start)/((float)CLOCKS_PER_SEC);
	/* execution statistics section */
	printf("*** level of parallelization ***\n");

	printf("*** matrix: %d,%d ***\n", A_size_x, B_size_y);
	printf("*** padded: %d,%d ***\n", A_size_p_x, B_size_p_y);
	printf("*** block: %d,%d ***\n", block_dim.x, block_dim.y);
	printf("*** thread: %d,%d ***\n", thread_dim.x, thread_dim.y);
// 	printf("*** threadc: %d ***\n", thread_dim.x * thread_dim.y);
	
	printf("*** time measurement ***\n");
	printf("*** CPU    : %f ms\n",cpu_time*1000);
	printf("*** GPU (a): %f ms\n",gpu_time/TIME_REP);
	printf("*** speedup: %f \n",cpu_time*1000/(gpu_time/TIME_REP));

// 	printf("%16.12f , %16.12f, %16.12f \n",cpu_time*1000,gpu_time, cpu_time*1000/gpu_time); // for batch execution
	
	/* controll section */

// 	display_matrix(A_size_x, B_size_y, matC);
// 	display_matrix(A_size_x, B_size_y, matC_dev);	
// 	int result= check_result(A_size_x, B_size_y, matC, matC_dev, false); // sampling
// 	int result= check_result_all(A_size_p_x, B_size_p_y, matC, matC_dev, 1e-3, 1); // full check
	int result= check_result_all_padd(A_size_x, B_size_y, A_size_p_x, B_size_p_y, matC, matC_dev, 5e-4, 1); // full check
	printf("-> result: %d <-\n", result);
	
	/* celanup section */
	free(matA);
	free(matB);
	free(matC);
	free(matC_dev);
	cudaFree(d_matA);
	cudaFree(d_matB);
	cudaFree(d_matC);

	return result;
}
