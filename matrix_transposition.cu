/*
 ============================================================================
 Name			: Fundamentals of GPU programming
 Author			: David Celny
 Date			: 01.01.2019
 Description	: transposition of matrix - Out of place + bandwidth measurement
 Tasks			: For case of 100 x 500 matrix do following:
				: Write a CPU matrix transpose implementation; calculate effective bandwidth.
 Check outputs of all your GPU kernels with it!
				: Write a row-to-row copy GPU kernel; use it to calculate effective bandwidth
				: Write GPU naïve matrix transpose kernel; calculate effective bandwidth
				: Write GPU transpose kernel with shared memory used to ensure the global
 memory read/write coalescence; calculate effective bandwidth
 Bonus Tasks	: Implement a variant where the input matrices are put into texture memory.
 				: use ILP for block methods
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
#define BLOCKNAIVE 1
#define SHARED 2
//#define BLOCKSHARED 3 // OBSOLETE
#define CACHED 4

#ifndef TRAN_VER
#define TRAN_VER SHARED // <- modify this to change operation
#endif // TRAN_VER

// TILE_SIZE is only valid for tiled
#ifndef TILE_SIZE
#define TILE_SIZE 16 //  <- modify this to change tiled approach size
#endif // TILE_SIZE

#ifndef ILP
#define ILP 4   // number of the work ech thread calculate
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
void matrix_transpose_c (unsigned int A_size_x, unsigned int A_size_y, float *matA, float *matB)
/*
 * the CPU implementation of matrix transposition
 * the matrixes are in flattened form -> index as vectors
 *   								  -> beware row/coll order
 */
{
	unsigned int i,j;

	for (i = 0; i < A_size_x; i++)
	{ // walk through rows of A matrix
		for (j = 0; j < A_size_y; j++)
		{ // walk through collumns of A matrix
			matB[j*A_size_x + i] = matA[i*A_size_y + j]; 
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

__device__ void submat_transpose_g_naive(unsigned int id_i,unsigned int id_j,float *matA, float *matB)
/*
 * the device function for result elemnet matrix transposition
 * expected the 2d call grid
 * general version with most uncoalesced( esp. matB writes) memory pattern 
 */
{	
	matB[id_j*d_A_size_x + id_i] = matA[id_i*d_A_size_y + id_j]; 
}

__device__ void submat_transpose_g_blocknaive(unsigned int id_i,unsigned int id_j,float *matA, float *matB)
/*
 * the device function for result elemnet matrix transposition
 * expected the 2d call grid with the y dimension is TILE_SIZE times smaller
 * tiled schema -> more execution per thread (in collumn direction)
 * general version with most uncoalesced( esp. matB writes) memory pattern 
 */
{	
#pragma unroll
	for(int k = 0; k< ILP; k++)
	{
		if (id_j + k > d_A_size_x) return; //if out of bounds 
		submat_transpose_g_naive( id_i, id_j+k, matA, matB);
	}
}

__device__ void submat_transpose_g_shared(unsigned int id_i, unsigned int id_j, float *tmp_A_submat,float *matA, float *matB)
/*
 * the device function for result block matrix transposition
 * expected the 2d thread grid
 * temporary storage for coalesced reads
 */
{		
	// !BEWARE changed perception of indexation and array definition
	// saved into already transposed form
	tmp_A_submat[threadIdx.x*TILE_SIZE + threadIdx.y] = matA[id_i*d_A_size_y + id_j];
	__syncthreads(); // to synch the loading into shared memory
	
	matB[id_j*d_A_size_x + id_i] = tmp_A_submat[threadIdx.x*TILE_SIZE + threadIdx.y];
}

// OBSOLETE if comparsion to transpose yield worse execution
// __device__ void submat_transpose_g_blockshared(unsigned int id_i, unsigned int id_j, float *tmp_A_submat, float *matA, float *matB)
// /*
//  * the device function for result block matrix transposition
//  * expected the 2d thread grid with the y dimension is TILE_SIZE/ILP
//  * tiled schema -> ILP more work per thread (in collumn direction)
//  * !!! BEWARE TILE_SIZE HAS TO BE DIVISIBLE BY ILP !!!
//  */
// {		
// 	// !BEWARE changed perception of indexation and array definition
// 	// saved into already transposed form	
// 	int k;
// 	for(k =0; k< ILP; k++)
// 	{
// 		if(id_j + k > d_A_size_x) break;
// 		tmp_A_submat[threadIdx.x*(ILP*TILE_SIZE) + threadIdx.y + k] = matA[id_i*d_A_size_y + id_j + k];	
// 	}
// 	__syncthreads(); // to synch the loading into shared memory
// 	
// 	for(k =0; k< ILP; k++)
// 	{
// 		if(id_j + k > d_A_size_x) return;
// 		matB[(id_j + k)*d_A_size_x + id_i] = tmp_A_submat[threadIdx.x*(ILP*TILE_SIZE) + threadIdx.y + k];
// 	}
// }

#if TRAN_VER == CACHED

texture<float, cudaTextureType1D, cudaReadModeElementType> texRef;

__device__ void submat_transpose_g_cached(unsigned int id_i, unsigned int id_j, float *tmp_A_submat, float *matB)
/*
 * the device function for result block matrix transposition
 * expected the 2d thread grid
 * temporary storage for coalesced reads
 */
{	
	
	tmp_A_submat[threadIdx.x*TILE_SIZE + threadIdx.y] = tex1Dfetch(texRef, id_i*d_A_size_y + id_j); 
	__syncthreads(); // to synch the loading into shared memory
	
	matB[id_j*d_A_size_x + id_i] = tmp_A_submat[threadIdx.x*TILE_SIZE + threadIdx.y];
}
#endif

__global__ void matrix_transpose_g (float *matA, float *matB)
/*
 * The kernel for matrix matrix multiplication
 * 	 expected to be called as two dimensional grid
 *   the matrixes are in flattened form -> indexed as vectors
 *   									-> beware row/coll order
 */
{
#if TRAN_VER == NAIVE
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
	
#elif TRAN_VER == BLOCKNAIVE
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
	id_y *= ILP; // to account for the ILP of each thread
#elif TRAN_VER == SHARED
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
	__shared__ float tmp_A_submat[TILE_SIZE*TILE_SIZE];
#elif TRAN_VER == BLOCKSHARED
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
	id_y *= ILP; // to account for the ILP of each thread
	__shared__ float tmp_A_submat[ILP*TILE_SIZE*TILE_SIZE];
#elif TRAN_VER == CACHED
	unsigned int id_x =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_y =  blockDim.y * blockIdx.y + threadIdx.y; 
	__shared__ float tmp_A_submat[TILE_SIZE*TILE_SIZE];
	
#endif

// 	if(id_x>= d_A_size_x) return;
// 	if(id_y>= d_A_size_y) return; // beware that the block version could overflow in x (row) dimension

#if TRAN_VER == NAIVE
	submat_transpose_g_naive (id_x, id_y, matA, matB);
#elif TRAN_VER == BLOCKNAIVE
	submat_transpose_g_blocknaive (id_x, id_y, matA, matB);
#elif TRAN_VER == SHARED
	submat_transpose_g_shared(id_x, id_y, tmp_A_submat, matA, matB);
#elif TRAN_VER == BLOCKSHARED
	submat_transpose_g_blockshared(id_x, id_y, tmp_A_submat, matA, matB);
#elif TRAN_VER == CACHED
	submat_transpose_g_cached(id_x, id_y, tmp_A_submat, matB);
#endif
}


__global__ void matrix_copy_g (float *matA, float *matB)
/*
 * The kernel for pure matrix copy
 * 	 expected to be called as two dimensional grid
 *	 used for effective bandwidth measurement
 *   the matrixes are in flattened form -> indexed as vectors
 *   									-> beware row/coll order
 */
{
	unsigned int id_i =  blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int id_j =  blockDim.y * blockIdx.y + threadIdx.y; 
	
	matB[id_i*d_A_size_y + id_j] = matA[id_i*d_A_size_y + id_j];
}

/* == the main call code == */

int main( int argc, char *argv[] )
/*
 * main function executing the kernell call and GPU properties output
 */
{
	const long int seed = 123456789;

	const unsigned int A_size_x = 16;
	const unsigned int A_size_y = 4;
	
	const unsigned int B_size_x = A_size_y;
	const unsigned int B_size_y = A_size_x;
		
	const unsigned int A_size_p_x = TILE_SIZE*(int)((A_size_x + TILE_SIZE -1)/TILE_SIZE);
	const unsigned int A_size_p_y = TILE_SIZE*(int)((A_size_y + TILE_SIZE -1)/TILE_SIZE);
	
	const unsigned int B_size_p_x = TILE_SIZE*(int)((B_size_x + TILE_SIZE -1)/TILE_SIZE);
	const unsigned int B_size_p_y = TILE_SIZE*(int)((B_size_y + TILE_SIZE -1)/TILE_SIZE);
	
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
	float *d_matB_b;
	
	// timing
	clock_t cpu_start, cpu_stop;
	cudaEvent_t gpu_start, gpu_start_b, gpu_stop, gpu_stop_b;
	float cpu_time, gpu_time, gpu_time_b;

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

	// the case of matrices without padding
// 	get_rnd_mat   (A_size_x, A_size_y, matA);
// 	get_zero_mat  (B_size_x, B_size_y, matB);
	
// 	display_matrix(A_size_x, A_size_y, matA); //DEBUG
// 	display_matrix(B_size_x, B_size_y, matB); //DEBUG
	
	get_rnd_mat_padd (A_size_x, A_size_y, A_size_p_x, A_size_p_y, matA);	
	get_zero_mat     (B_size_p_x, B_size_p_y, matB);	

// 	display_matrix (A_size_p_x, A_size_p_y, matA); //DEBUG
// 	display_matrix (B_size_p_x, B_size_p_y, matB); //DEBUG

	// the device init
	cudaSafeCall(cudaMalloc((void**)&d_matA, A_size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matB, B_size_n));
	cudaSafeCall(cudaMalloc((void**)&d_matB_b, A_size_n)); // for the bandwidth check
	
	// constant memory move
	cudaSafeCall(cudaMemcpyToSymbol(d_A_size_x, &A_size_p_x, sizeof(unsigned int)));
	cudaSafeCall(cudaMemcpyToSymbol(d_A_size_y, &A_size_p_y, sizeof(unsigned int)));
	
	// the data move host->device
	cudaSafeCall(cudaMemcpy(d_matA, matA, A_size_n, cudaMemcpyHostToDevice));
	
	// cached variant initialisation
#if TRAN_VER == CACHED
// 	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray* cuArray;
	cudaMalloc(&cuArray, A_size_n);

	cudaMemcpy(cuArray, matA, A_size_n, cudaMemcpyHostToDevice);
	
	texRef.addressMode[0]	= cudaAddressModeClamp;
	texRef.addressMode[1]	= cudaAddressModeClamp;
	texRef.filterMode		= cudaFilterModePoint;
	texRef.normalized		= false;
	
	size_t offset = 0;
	cudaBindTexture(&offset, texRef, cuArray, A_size_n);
	
#endif
	
	// timing
	cudaSafeCall(cudaEventCreate(&gpu_start));
	cudaSafeCall(cudaEventCreate(&gpu_stop));
	
	cudaSafeCall(cudaEventCreate(&gpu_start_b));
	cudaSafeCall(cudaEventCreate(&gpu_stop_b));
	
	/* Kernell call section */
#if TRAN_VER == NAIVE || TRAN_VER == SHARED || TRAN_VER == CACHED
	dim3 thread_dim(TILE_SIZE,TILE_SIZE);
	dim3 block_dim(A_size_p_x/TILE_SIZE, A_size_p_y/TILE_SIZE);
#else  
	dim3 thread_dim(TILE_SIZE,TILE_SIZE);	
	dim3 block_dim(A_size_p_x/TILE_SIZE, ceil((float)A_size_p_y/TILE_SIZE/ILP)); //ILP in A_y case
#endif
		
	/* to get the device initialisation in case that is not measured*/
	matrix_copy_g<<<block_dim,thread_dim>>>(d_matA, d_matB_b);
	cudaDeviceSynchronize();
	
	/* the copy kernel */
	
	cudaSafeCall(cudaEventRecord(gpu_start_b, 0));
	
	matrix_copy_g<<<block_dim,thread_dim>>>(d_matA, d_matB_b);

	cudaDeviceSynchronize();
	cudaSafeCall(cudaEventRecord(gpu_stop_b, 0));
	cudaSafeCall(cudaEventSynchronize(gpu_stop_b));
//	cudaSafeKernell(); // not to be included in speed measurement

	cudaSafeCall(cudaEventElapsedTime(&gpu_time_b,gpu_start_b,gpu_stop_b));
	
	/* the transpose kernel */
	cudaSafeCall(cudaEventRecord(gpu_start, 0));
	
#if TRAN_VER >= NAIVE && TRAN_VER <= CACHED // positive integer in the range naive till cached
	matrix_transpose_g<<<block_dim,thread_dim>>>(d_matA, d_matB);
#else
	printf(" !!! Unsuported version option: %d !!! \n",TRAN_VER);
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

	matrix_transpose_c(A_size_p_x, A_size_p_y, matA, matB);

	cpu_stop = clock();

	cpu_time = (cpu_stop-cpu_start)/((float)CLOCKS_PER_SEC);
	/* execution statistics section */
// 	printf("*** level of parallelization ***\n");

	printf("*** matrix: %d,%d ***\n", A_size_x, A_size_y);
	printf("*** padded: %d,%d ***\n", A_size_p_x, B_size_p_y);
	printf("*** block: %d,%d ***\n", block_dim.x, block_dim.y);
	printf("*** thread: %d,%d ***\n", thread_dim.x, thread_dim.y);
	
	printf("*** time measurement ***\n");
	printf("*** CPU: %f ms\n",cpu_time*1000);
	printf("*** GPU_bench: %f ms\n",gpu_time_b);
	printf("*** GPU: %f ms\n",gpu_time);	
	printf("*** speedup: %f \n",cpu_time*1000/gpu_time);
	printf("*** bandwidth_bench: %f \n",2*A_size_n/gpu_time_b/1024/1024/1024); // with 3* '/1024' in GB
	printf("*** bandwidth: %f \n",2*A_size_n/gpu_time/1024/1024/1024); // with 3* '/1024' in GB

// 	printf("%16.12f , %16.12f, %16.12f \n",cpu_time*1000,gpu_time, cpu_time*1000/gpu_time); // for batch execution
	
	/* controll section */

	// display_matrix(B_size_p_x, B_size_p_y, matB);
	// display_matrix(B_size_p_x, B_size_p_y, matB_dev);
// 	int result= check_result(B_size_x, B_size_y, matB, matB_dev, false); // sampling
// 	int result= check_result_all(B_size_x, B_size_y, matB, matB_dev, 1e-6, true); // full check for non padded matrix
	int result= check_result_all_padd(B_size_x, B_size_y, B_size_p_x, B_size_p_y, matB, matB_dev, 1e-6, true); // full check for padded matrix
	
	/* celanup section */
	free(matA);
	free(matB);
	free(matB_dev);
	cudaFree(d_matA);
	cudaFree(d_matB);
	cudaFree(d_matB_b);

	return result;
}
