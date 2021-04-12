/*
 * Author: Shubham Mittal
 * IIITD, 2018101
 * GPU Project: Accelerating Concurrent Heap on GPUs
*/
#include "td_ins-td_del.h"

__device__ int get_lock_state(int node_idx, 
                              int *heap_locks)
{
    return heap_locks[node_idx];
}

__device__ void take_lock(int *lock, 
                          int lock_state_1, 
                          int lock_state_2)
{
    while (atomicCAS(lock, lock_state_1, lock_state_2) != lock_state_1);
}

__device__ int try_lock(int *lock, 
                        int lock_state_1, 
                        int lock_state_2)
{
    return atomicCAS(lock, lock_state_1, lock_state_2);
}

__device__ void release_lock(int *lock, 
                             int lock_state_1, 
                             int lock_state_2)
{
    atomicCAS(lock, lock_state_1, lock_state_2);
}

__device__ void bitonic_sort(int *arr, 
                             int size)
{
    // assuming size = power of 2
    int my_thread_id = threadIdx.x;
    int maximum = 0, minimum = 0, other_idx = 0, i = 2, j = 2;
    // parity of batch number will tell which operation to perform in that batch:
    // min/max for even/odd respectively
    int my_batch_number = my_thread_id >> 1;
    for (i = 2; i <= size ; i <<= 1, my_batch_number >>= 1)
    {
        for(j = i; j >= 2 ; j >>= 1)
        {
            int steps_to_look_ahead = j >> 1;
            // only first half of any batch can be active
            if ((my_thread_id % j < steps_to_look_ahead) && (my_thread_id < size)) 
            {
                other_idx = my_thread_id + steps_to_look_ahead;
                minimum = min(arr[my_thread_id], arr[other_idx]);
                maximum = max(arr[my_thread_id], arr[other_idx]); // chances of improvement by using minimum to find max
                if(my_batch_number & 1)
                {
                    arr[my_thread_id] = maximum;
                    arr[other_idx] = minimum;
                }
                else
                {
                    arr[my_thread_id] = minimum;
                    arr[other_idx] = maximum;
                }
            }
            __syncthreads();
        }
    }
}


__device__ void copy_shared_to_global(int *shared, int from_shared_idx1, int to_shared_idx2, int *global, int from_global_idx1)
{
    int my_thread_id = threadIdx.x;
    int n = to_shared_idx2 - from_shared_idx1;
    if(my_thread_id < n)
    {
        global[from_global_idx1 + my_thread_id] = shared[from_shared_idx1 + my_thread_id];
    }
    __syncthreads();
}

__device__ void copy_global_to_shared(int *global, int from_global_idx1, int to_global_idx2, int *shared, int from_shared_idx1)
{
    int my_thread_id = threadIdx.x;
    int n = to_global_idx2 - from_global_idx1;
    if(my_thread_id < n)
    {
        shared[from_shared_idx1 + my_thread_id] = global[from_global_idx1 + my_thread_id];
    }
    __syncthreads();
}

__global__ void heap_init(Heap *heap, 
                          Partial_Buffer *partial_buffer)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index < HEAP_CAPACITY)
    {
        heap -> arr[index] = INT_MAX;
    }
    if(index < PARTIAL_BUFFER_CAPACITY)
    {
        partial_buffer -> arr[threadIdx.x] = INT_MAX;   
    }
}

__global__ void td_insertion(int *items_to_be_inserted, 
                             int number_of_items_to_be_inserted, 
                             int *heap_locks, 
                             Partial_Buffer *partial_buffer,
                            Heap *heap)
{
    /*
     * number_of_items_to_be_inserted <= BATCH_SIZE
    */
    __shared__ int items_to_be_inserted_shared_mem[BATCH_SIZE];
    __shared__ int array_to_be_merged_shared_mem[BATCH_SIZE];
    __shared__ int merged_array_shared_mem[BATCH_SIZE << 1];

    int my_thread_id = threadIdx.x;
    copy_global_to_shared(items_to_be_inserted, 0, number_of_items_to_be_inserted, items_to_be_inserted_shared_mem, 0);
    
    bitonic_sort(items_to_be_inserted_shared_mem, number_of_items_to_be_inserted);

    int root_node_idx = 0;
    if (my_thread_id == 0)
        take_lock(&heap_locks[root_node_idx], AVAILABLE, INUSE);

    if (partial_buffer -> size + number_of_items_to_be_inserted >= BATCH_SIZE)
    {

    }
    else
    {
        release_lock(&heap_locks[root_node_idx], AVAILABLE, INUSE);
        // return;
    }
    if (my_thread_id < number_of_items_to_be_inserted)
    {
        items_to_be_inserted[my_thread_id] = items_to_be_inserted_shared_mem[my_thread_id] ;
    }
}

__host__ void heap_init()
{
    gpuErrchk( cudaMalloc(&d_partial_buffer, sizeof(Partial_Buffer)));
    gpuErrchk( cudaMalloc(&d_heap, sizeof(Heap))); // need to fill with INT_MAX

    heap_init<<<ceil(HEAP_CAPACITY/1024.0), 1024>>>(d_heap, d_partial_buffer);

    gpuErrchk( cudaMalloc((void**)&d_heap_locks, NUMBER_OF_NODES * sizeof(int)) );
    gpuErrchk( cudaMemset(d_heap_locks, AVAILABLE, NUMBER_OF_NODES * sizeof(int)) );
}