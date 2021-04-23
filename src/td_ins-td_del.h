/*
 * Author: Shubham Mittal
 * IIITD, 2018101
 * GPU Project: Accelerating Concurrent Heap on GPUs
*/
#include "td_ins-td_del-runtime.h"
/* 
 * BLOCK_SIZE >= BATCH_SIZE always to ensure correctness of code
 * For full usage of GPU resources, BLOCK_SIZE = BATCH_SIZE 
 * BATCH_SIZE should be power of 2
 */

#define BLOCK_SIZE BATCH_SIZE
#define PARTIAL_BUFFER_CAPACITY (BATCH_SIZE - 1)
// node 0 will be wasted
#define HEAP_CAPACITY (NUMBER_OF_NODES + 1) * (BATCH_SIZE) 
#define ROOT_NODE_IDX 1
#define MASTER_THREAD 0

enum LOCK_STATES {AVAILABLE, INUSE};

struct Partial_Buffer
{
    int size = 0;
    int arr[PARTIAL_BUFFER_CAPACITY];
};

struct Heap
{
    int size = 0;
    int global_id = 1;  // helps in taking root lock in sequential order of kernel invocations
    int arr[HEAP_CAPACITY];
};

Heap *d_heap;
Partial_Buffer *d_partial_buffer;
int *d_heap_locks;
cudaStream_t stream[NUMBER_OF_CUDA_STREAMS];
int kernel_id = 1;
int stream_id = 0;

// lock related APIs implemented using atomicCAS
__device__ int get_lock_state(int node_idx, int *heap_locks);
__device__ void take_lock(int *lock, int lock_state_1, int lock_state_2);
__device__ int try_lock(int *lock, int lock_state_1, int lock_state_2);
__device__ void release_lock(int *lock, int lock_state_1, int lock_state_2);
// initialise heap
__global__ void heap_init(Heap *heap,  Partial_Buffer *partial_buffer);
// return bit-reversal value of n
__device__ int bit_reversal(int n);
// copies arr1 into arr2
__device__ void copy_arr1_to_arr2(int *arr1, int from_arr1_idx1, int *arr2, int from_arr2_idx1, int num_of_elements);
// sets values of arr1 to a particular value
__device__ void memset_arr(int *arr, int from_arr_idx1, int val, int num_of_elements);
// Algorithm referenced from https://wiki.rice.edu/confluence/download/attachments/4435861/comp322-s12-lec28-slides-JMC.pdf?version=1&modificationDate=1333163955158
__device__ void bitonic_sort(int *arr, int size);
// returns correct index of the searched element in the final merged array
__device__ int binary_search(int *arr1, int high, int search, bool consider_equality);
// https://www2.hawaii.edu/~nodari/teaching/f16/notes/notes10.pdf
// merge two sorted lists
__device__ void merge_and_sort(int *arr1, int idx1, int *arr2, int idx2, int *merged_arr);
// insert a node of batch
__global__ void td_insertion(int *items_to_be_inserted, int number_of_items_to_be_inserted, int *heap_locks, Partial_Buffer *partial_buffer, Heap *heap, int my_id);
// delete a node of batch from heap
__global__ void td_delete(int *items_deleted, int *heap_locks, Partial_Buffer *partial_buffer, Heap *heap, int my_id);
__host__ int get_kernel_id();
__host__ void next_stream_id();
__host__ cudaStream_t get_current_stream();
