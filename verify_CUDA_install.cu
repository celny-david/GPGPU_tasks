/*
 ============================================================================
 Name			: Fundamentals of GPU programing
 Author			: David Celny
 Date			: 14.10.2018
 Description	: initial verification of the CUDA installation
 ============================================================================
 */

#include <stddef.h>
#include <stdio.h>

/* === Error checking utility section === */
#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaSafeKernell()    __cudaCheckError( __FILE__, __LINE__ )

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
__global__ void say_hello()
/*
 * The kernel for print of Hello World message
 * expected to be called with 1D thread distribution
 */
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	printf("Hello world from thread with index %d! \n", idx);
}

int main()
/*
 * main function executing the kernell call and GPU properties output
 */
{	
	const int thread_count = 16; // number of threads used per block
	const int block_count = 2; // number of blocks used 
	
	int awailable_device; // the device number ! expect only single GPU coprocessor presence to work as intended
		
	cudaDeviceProp device_prop;
	cudaSafeCall( cudaGetDevice(&awailable_device)); //get the device count
	cudaSafeCall( cudaGetDeviceProperties(&device_prop, awailable_device)); // get the last device properties
	
	/* Device infromation print section */
	printf("*** Hello world from %s coprocessor ***\n", device_prop.name);
	printf("*** SM: %i, Gmem: %3.2f GB, Smem/B: %zu kB, Cmem: %zu kB ***\n",(int)device_prop.multiProcessorCount
																	  ,(float)((size_t)device_prop.totalGlobalMem)/(2<<29)
																	  ,(size_t)device_prop.sharedMemPerBlock/(2<<9)
																	  ,(size_t)device_prop.totalConstMem/(2<<9));
	printf("\n");
	printf("The hello kernel is invoked with following <<< %d, %d >>> \n", block_count, thread_count);
	
	/* Kernell call section */
	
	say_hello<<<block_count,thread_count>>>();
	cudaDeviceSynchronize();
	cudaSafeKernell();
	
	return 0;
}