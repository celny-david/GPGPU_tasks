# Info
Set of tasks for programming for [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units) using CUDA. Task presented can be configured in code to produce varying metrics output. An option for batch execution is also possible in relevant cases.
Showcase parallel matrix operations like addition, multiplication, transposition in comparison with CPU and different parallel aproaches.

# Content and its capabilities
 - verify_CUDA_install.cu
 	- show the initial setup for kernel launch
 	- show the simple print of GPU properties
 	- include solution for error forward from CUDA

 - matrix_addition.cu
 	- sequential and parallel implementation of the matrix addition
 	- for rectangular matrices of arbitrary size
 	- include verification of correctness between CPU, GPU result

 - instruction_level_paralellism.cu
 	- sequential and parallel implementation of the elementwise matrix addition and multiplication
 	- demonstrate the effects of Instruction-level-parallelism (ILP)

 - matrix_multiplication.cu
 	- sequential and parallel rectangular matrix multiplication implementation
 		- matrices are represented as flattened to 1d vector
 	- showcasing three approaches Naive, Shared memory, Cached in texture memory
 	- implemented a variant with matrix padding
 	- include verification of correctness between CPU, GPU result

 - matrix_transposition.cu
 	- sequential and parallel rectangular matrix transposition implementation
 	- showcasing approaches: Naive, Block approach, with Shared memory, Cached in texture memory
 	- include use of ILP in block approach
 	- include bandwidth calculation and comparison
 	- include verification of correctness between CPU, GPU result

 - parallel_reduction.cu
 	- sequential and parallel implementation of vector reduction
 	- showcasing three approaches: Atomic, Cascade and Harris algorithm
 	- include use of ILP in cascade
 	- include bandwidth calculation and comparison
 	- include verification of correctness between CPU, GPU result

## compile & run
 - all tasks can be compiled with `nvcc <name-of-the-task>.cu`
 	- beware the issue with use of texture memory for combination of nvcc 12.3 and gcc 13.2
 	- please use different combination where this issue is repaired for compulation of the CACHED cases
 	- when batch processing is desired compile task variation with different values for macro values i.e. using `-DILP=16`
 	- most of the define macros are configurable from cli to enable manipulation of matrix size and other configuration and algorithm variant selection. See source code for specific guidance.
 - programs can be simply run with `./a.out`
 	- if configured (commented in the code) one can utilize shorter prints for batch execution

# Concluding remarks
This is a showcase primarily intended for the purposes of portfolio, but still i would like to preserve certain educational angle of the showcased tasks. This can be used as inspiration and learning material for those starting with GPGPU and takcling the tricky linear argebra tasks usually used for testing.
If you find any issue with the the tasks plese do not hesitate to report it. Also be aware that this implementation was primarily developed on/for Linux.
