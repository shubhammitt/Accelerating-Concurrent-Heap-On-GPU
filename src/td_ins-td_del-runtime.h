#include<iostream>

#define BATCH_SIZE 1024
 // should be power of 2
#define NUMBER_OF_NODES (1<<17)
// should be <= 20, depends on the number of SMs in a GPU
#define NUMBER_OF_CUDA_STREAMS 16


// Code taken from https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ void heap_init();
__host__ void insert_keys(int *items_to_be_inserted, int total_num_of_keys_insertion);
__host__ void delete_keys(int *items_to_be_deleted);
__host__ void heap_finalise();