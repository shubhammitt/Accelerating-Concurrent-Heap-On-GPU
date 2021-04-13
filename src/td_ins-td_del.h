#include<iostream>
using namespace std;

/* 
 * ALERT: BLOCK_SIZE >= BATCH_SIZE always to ensure correctness of code
 * For full usage of GPU resources, BLOCK_SIZE = BATCH_SIZE 
 * BATCH_SIZE should be power of 2
 */

#define BATCH_SIZE 1024
#define BLOCK_SIZE BATCH_SIZE
#define PARTIAL_BUFFER_CAPACITY (BATCH_SIZE - 1)
#define NUMBER_OF_NODES (1<<19 - 1)
#define HEAP_CAPACITY NUMBER_OF_NODES * BATCH_SIZE
#define ROOT_NODE_IDX 1
#define MASTER_THREAD 0

enum LOCK_STATES {AVAILABLE, INUSE, TARGET, MARKED};

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

struct Partial_Buffer
{
    int size = 0;
    int arr[PARTIAL_BUFFER_CAPACITY];
};

struct Heap
{
    int size = 0;
    int arr[HEAP_CAPACITY];
};

#ifndef GLOBAL_H
#define GLOBAL_H

inline Heap *d_heap;
inline Partial_Buffer *d_partial_buffer;
inline int *d_heap_locks;

#endif

__device__ int get_lock_state(int node_idx, int *heap_locks);
__device__ void take_lock(int *lock, int lock_state_1, int lock_state_2);
__device__ int try_lock(int *lock, int lock_state_1, int lock_state_2);
__device__ void release_lock(int *lock, int lock_state_1, int lock_state_2);

__global__ void heap_init(Heap *heap,  Partial_Buffer *partial_buffer);
__device__ int bit_reversal(int n);
__device__ void copy_arr1_to_arr2(int *arr1, int from_arr_idx1, int to_arr_idx2, int *arr2, int from_arr2_idx1);
__device__ void memset_arr(int *arr, int from_arr_idx1, int to_arr_idx2, int val = INT_MAX);
// Algorithm referenced from https://wiki.rice.edu/confluence/download/attachments/4435861/comp322-s12-lec28-slides-JMC.pdf?version=1&modificationDate=1333163955158
__device__ void bitonic_sort(int *arr, int size);
__device__ int binary_search(int *arr1, int high, int search, bool consider_equality);
// https://www2.hawaii.edu/~nodari/teaching/f16/notes/notes10.pdf
__device__ void merge_and_sort(int *arr1, int idx1, int *arr2, int idx2, int *merged_arr);
__global__ void td_insertion(int *items_to_be_inserted, int number_of_items_to_be_inserted, int *heap_locks, Partial_Buffer *partial_buffer, Heap *heap);

__host__ void heap_init();
