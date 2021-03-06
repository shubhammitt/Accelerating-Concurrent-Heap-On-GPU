/*
 * Author: Shubham Mittal
 * IIITD, 2018101
 * GPU Project: Accelerating Concurrent Heap on GPUs
*/
#include "td_ins-td_del.h"


__device__ int get_lock_state(int node_idx, int *heap_locks) {
    // get the state of lock at node_idx
    return heap_locks[node_idx];
}


__device__ void take_lock(int *lock, int lock_state_1, int lock_state_2) {
    // take lock and loop until successful
    while (atomicCAS(lock, lock_state_1, lock_state_2) != lock_state_1);
}


__device__ int try_lock(int *lock, int lock_state_1, int lock_state_2) {
    // try to take lock and exit
    return atomicCAS(lock, lock_state_1, lock_state_2);
}


__device__ void release_lock(int *lock, int lock_state_1, int lock_state_2) {
    // release lock atmically
    atomicCAS(lock, lock_state_1, lock_state_2);
}


__global__ void heap_init(Heap *heap, Partial_Buffer *partial_buffer) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index == 0) {
        heap -> global_id = 1;
        heap -> size = 0;
        partial_buffer -> size = 0;
    }

    // initialise heap values and partial buffer to INT_MAX
    if (index < HEAP_CAPACITY) {
        heap -> arr[index] = INT_MAX;
    }

    if (index < PARTIAL_BUFFER_CAPACITY) {
        partial_buffer -> arr[index] = INT_MAX;   
    }
}


__device__ int bit_reversal(int n, int level) {
    // reverse all bits but first

    if (n <= 4) // base case
        return n;

    int ans = 1 << (level--);
    while(n != 1) {
        ans += ((n & 1) << (level--));
        n >>= 1;
    }
    return ans;
}


__device__ void copy_arr1_to_arr2(int *arr1, int from_arr1_idx1, int *arr2, int from_arr2_idx1, int num_of_elements) {
    int my_thread_id = threadIdx.x;
    if (my_thread_id < num_of_elements) {
        arr2[from_arr2_idx1 + my_thread_id] = arr1[from_arr1_idx1 + my_thread_id];
    }
    __syncthreads();
}


__device__ void memset_arr(int *arr, int from_arr_idx1, int val, int num_of_elements) {
    // sets values of arr to val between given indices of arr
    int my_thread_id = threadIdx.x;
    if (my_thread_id < num_of_elements) {
        arr[from_arr_idx1 + my_thread_id] = val;
    }
    __syncthreads();
}

__device__ void bitonic_sort(int *arr, int size) {
    // assuming size = power of 2
    int my_thread_id = threadIdx.x;
    int maximum = 0, minimum = 0, other_idx = 0, i = 2, j = 2;
    // parity of batch number will tell which operation to perform in that batch:
    // min/max for even/odd respectively
    int my_batch_number = my_thread_id >> 1;
    for (i = 2; i <= size ; i <<= 1, my_batch_number >>= 1) {
        for (j = i; j >= 2 ; j >>= 1) {
            int steps_to_look_ahead = j >> 1;
            // only first half of any batch can be active
            if ((my_thread_id % j < steps_to_look_ahead) && (my_thread_id < size)) 
            {
                other_idx = my_thread_id + steps_to_look_ahead;
                minimum = min(arr[my_thread_id], arr[other_idx]);
                maximum = max(arr[my_thread_id], arr[other_idx]); // chances of improvement by using minimum to find max
                if (my_batch_number & 1) {
                    arr[my_thread_id] = maximum;
                    arr[other_idx] = minimum;
                }
                else {
                    arr[my_thread_id] = minimum;
                    arr[other_idx] = maximum;
                }
            }
            __syncthreads();
        }
    }
}


__device__ int binary_search(int *arr1, int high, int search, bool consider_equality) {
    /*
     * Finds index of the smallest element larger than the element searched in arr1
     * and consider equality means equal element will not be considered larger 
    */
    if(high == 0) return 0;
    int low = 0, mid = 0;
    int ans = high;
    while (low <= high)
    {
        // consider middle index between low and high
        mid = (low + high) >> 1;
        if (arr1[mid] >= search and consider_equality) {
            // search in left half
            ans = mid;
            high = mid - 1;
        }
        else if (arr1[mid] > search) {
            // search in left half
            ans = mid;
            high = mid - 1;
        }
        else {
            // search in right half
            low = mid + 1;
        }
    }
    return ans;
}


__device__ void merge_and_sort(int *arr1, int idx1, int *arr2, int idx2, int *merged_arr) {

    __syncthreads();
    if(idx1 == 0) {
        // arr1 if empty then simply copy arr2 
        copy_arr1_to_arr2(arr2, 0, merged_arr, 0, idx2);
    }
    else if(idx2 == 0) {
        // arr2 if empty then simply copy arr1
        copy_arr1_to_arr2(arr1, 0, merged_arr, 0, idx1);
    }
    
    else if(arr1[idx1 - 1] <= arr2[0]) {
        // all elements of arr1 <= all elements of arr2
        copy_arr1_to_arr2(arr1, 0, merged_arr, 0, idx1);
        copy_arr1_to_arr2(arr2, 0, merged_arr, idx1, idx2);
    }

    else if(arr2[idx2 - 1] <= arr1[0]) {\
        // all elements of arr2 <= all elements of arr1
        copy_arr1_to_arr2(arr2, 0, merged_arr, 0, idx2);
        copy_arr1_to_arr2(arr1, 0, merged_arr, idx2, idx1);
    }
    else {
        // no special case so need to perform binary search
        int my_thread_id = threadIdx.x;
        if (my_thread_id < idx1) {
            int x = binary_search(arr2, idx2, arr1[my_thread_id], 1);
            merged_arr[my_thread_id + x] = arr1[my_thread_id];
        }

        if (my_thread_id < idx2) {
            int x = binary_search(arr1, idx1, arr2[my_thread_id], 0);
            merged_arr[my_thread_id + x] = arr2[my_thread_id];
        }
    }
    __syncthreads();
    
}


__global__ void td_insertion(int *items_to_be_inserted, int number_of_items_to_be_inserted, int *heap_locks, Partial_Buffer *partial_buffer, Heap *heap, int my_kernel_id) {
    /*
     * number_of_items_to_be_inserted <= BATCH_SIZE
    */
    int my_thread_id = threadIdx.x;

    __shared__ int items_to_be_inserted_shared_mem[BATCH_SIZE];
    __shared__ int array_to_be_merged_shared_mem[BATCH_SIZE];
    __shared__ int merged_array_shared_mem[BATCH_SIZE << 1];

    // copy keys to be inserted in shared memory
    copy_arr1_to_arr2(items_to_be_inserted, 0, items_to_be_inserted_shared_mem, 0, number_of_items_to_be_inserted);

    // sort the keys to be inserted
    bitonic_sort(items_to_be_inserted_shared_mem, number_of_items_to_be_inserted);

     // take root node lock
     if (my_thread_id == MASTER_THREAD){
        while(atomicCAS(&(heap -> global_id), my_kernel_id, 0) != my_kernel_id);
        take_lock(&heap_locks[ROOT_NODE_IDX], AVAILABLE, INUSE);
        atomicCAS(&(heap -> global_id), 0, my_kernel_id + 1);
    }
    __syncthreads();
    
    // copy partial buffer into shared memory
    copy_arr1_to_arr2(partial_buffer -> arr, 0, array_to_be_merged_shared_mem, 0, partial_buffer -> size);

    int combined_size = partial_buffer -> size + number_of_items_to_be_inserted;
    // merge partial buffer and keys to be inserted
    merge_and_sort(items_to_be_inserted_shared_mem, number_of_items_to_be_inserted, \
            array_to_be_merged_shared_mem, partial_buffer -> size, merged_array_shared_mem);
    
    if (combined_size >= BATCH_SIZE) {

        // copy batch_size into insertion key list
        copy_arr1_to_arr2(merged_array_shared_mem, 0, items_to_be_inserted_shared_mem, 0, BATCH_SIZE);

        // copy rest over in partial buffer and update its size
        copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE , partial_buffer -> arr, 0, combined_size - BATCH_SIZE);
        __threadfence();

        // update partial buffer size
        if (my_thread_id == MASTER_THREAD)
            atomicExch(&(partial_buffer -> size), combined_size - BATCH_SIZE);
        __syncthreads();
    }
    else {
        if (heap -> size == 0) {
            // transfer all keys in partial buffer
            copy_arr1_to_arr2(merged_array_shared_mem, 0, partial_buffer -> arr, 0, combined_size);
            __threadfence();
        }
        else {
            // copy partial buffer into shared array
            copy_arr1_to_arr2(merged_array_shared_mem, 0, array_to_be_merged_shared_mem, 0, combined_size);
            
            // copy root node into shared memory
            copy_arr1_to_arr2(heap -> arr, ROOT_NODE_IDX * BATCH_SIZE, items_to_be_inserted_shared_mem, 0, BATCH_SIZE);
            __syncthreads();

            // merge partial buffer with root node
            merge_and_sort(items_to_be_inserted_shared_mem, BATCH_SIZE, array_to_be_merged_shared_mem, combined_size, merged_array_shared_mem);
        
            // copy back to root node
            copy_arr1_to_arr2(merged_array_shared_mem, 0, heap -> arr, ROOT_NODE_IDX * BATCH_SIZE, BATCH_SIZE);
            __threadfence();

            // copy to partial buffer
            copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, partial_buffer -> arr, 0, combined_size);
            __threadfence();
        }
        // update partial buffer size
        if (my_thread_id == MASTER_THREAD)
            partial_buffer -> size = combined_size;

        __syncthreads();

        if (my_thread_id == MASTER_THREAD)
            release_lock(&heap_locks[ROOT_NODE_IDX], INUSE, AVAILABLE);
        return;
    }

    // increase the heap size by 1 atomically
    if (my_thread_id == MASTER_THREAD)
        atomicAdd(&(heap -> size), 1);
    __syncthreads();

    int tar = heap -> size, level = -1;
    int dummy_tar = tar;
    while(dummy_tar) {
        level++;
        dummy_tar >>= 1;
    }

    tar = bit_reversal(tar, level);
    
    // take lock on target node 
    if (tar != ROOT_NODE_IDX) {
        if (my_thread_id == MASTER_THREAD) {
            take_lock(&heap_locks[tar], AVAILABLE, INUSE);
        }
        __syncthreads();
    }

    int low = 0, cur = ROOT_NODE_IDX;;
    while (cur != tar) {
        low = cur * BATCH_SIZE;
        // copy current node to shared mem
        copy_arr1_to_arr2(heap -> arr, low, array_to_be_merged_shared_mem, 0, BATCH_SIZE);

        // merger current batch with insertion list in shared mem
        merge_and_sort(array_to_be_merged_shared_mem, BATCH_SIZE, items_to_be_inserted_shared_mem, BATCH_SIZE, merged_array_shared_mem);

        // copy back to current batch
        copy_arr1_to_arr2(merged_array_shared_mem, 0, heap -> arr, low, BATCH_SIZE);
        __threadfence();

        // copy to insertion list
        copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, items_to_be_inserted_shared_mem, 0, BATCH_SIZE);

        // go to child node
        cur = tar >> (--level);

        if(my_thread_id == MASTER_THREAD) {
            if (cur != tar) {
                // take lock of child node
                take_lock(&heap_locks[cur], AVAILABLE, INUSE);
            }
            // release lock of parent node
            release_lock(&heap_locks[cur >> 1], INUSE, AVAILABLE);
        }
        __syncthreads();
    }

    // copy elements to be inserted at target node
    copy_arr1_to_arr2(items_to_be_inserted_shared_mem, 0, heap -> arr, tar * BATCH_SIZE, BATCH_SIZE);
    __threadfence();
    if(my_thread_id == MASTER_THREAD) {
        release_lock(&heap_locks[tar], INUSE, AVAILABLE);
    }
    __syncthreads();
}

__global__ void td_delete(int *items_deleted, int *heap_locks, Partial_Buffer *partial_buffer, Heap *heap, int my_kernel_id) {
    
    int my_thread_id = threadIdx.x;

    __shared__ int arr1_shared_mem[BATCH_SIZE];
    __shared__ int arr2_shared_mem[BATCH_SIZE];
    __shared__ int arr3_shared_mem[BATCH_SIZE];
    __shared__ int merged_array_shared_mem[BATCH_SIZE << 2];

    // take root node lock
    if (my_thread_id == MASTER_THREAD)
    {
        while(atomicCAS(&(heap -> global_id), my_kernel_id, 0) != my_kernel_id);
        take_lock(&heap_locks[ROOT_NODE_IDX], AVAILABLE, INUSE);
        atomicCAS(&(heap -> global_id), 0, my_kernel_id + 1);
    }
    __syncthreads();

    // heap is empty
    if (heap -> size == 0) {
        // if partial buffer is not empty then copy it into list of deleted items
        if (partial_buffer -> size != 0) {
            copy_arr1_to_arr2(partial_buffer -> arr, 0, items_deleted, 0, partial_buffer -> size);
            __threadfence();
            if(my_thread_id == MASTER_THREAD) {
                atomicExch(&(partial_buffer -> size), 0);
            }
            __syncthreads();
        }
        // release root node lock
        if (my_thread_id == MASTER_THREAD) {
            release_lock(&heap_locks[ROOT_NODE_IDX], INUSE, AVAILABLE);
        }
        __syncthreads();
        return;
    }

    // copy root node elements into items deleted and replace them with max value
    copy_arr1_to_arr2(heap -> arr, ROOT_NODE_IDX * BATCH_SIZE, items_deleted, 0, BATCH_SIZE);
    memset_arr(heap -> arr, ROOT_NODE_IDX * BATCH_SIZE, INT_MAX, BATCH_SIZE);
    __threadfence();

    int tar = heap -> size, level = -1; // may have floating point error, default to -1
    int dummy_tar = tar;
    while(dummy_tar) {
        level++;
        dummy_tar >>= 1;
    }
    tar = bit_reversal(tar, level);

    __syncthreads(); // necessary so that master thread do not decrement while other threads are initialising tar
    if (my_thread_id == MASTER_THREAD) {
        atomicAdd(&(heap -> size), -1);
    }
    __syncthreads();

    if(tar == 1) {
        // root node is target so no need to proceed further
        if(my_thread_id == MASTER_THREAD)
            release_lock(&heap_locks[ROOT_NODE_IDX], INUSE, AVAILABLE);
        __syncthreads();
        return; 
    }

    // take lock on target
    if (my_thread_id == MASTER_THREAD) 
        take_lock(&heap_locks[tar], AVAILABLE, INUSE);
    __syncthreads();
    
    // copy elements from target to root and reset target
    copy_arr1_to_arr2(heap -> arr, tar * BATCH_SIZE, heap -> arr, ROOT_NODE_IDX * BATCH_SIZE, BATCH_SIZE);
    __threadfence();
    memset_arr(heap -> arr, tar * BATCH_SIZE, INT_MAX, BATCH_SIZE);
    __threadfence();
    
    // release lock on target
    if (my_thread_id == MASTER_THREAD) 
        release_lock(&heap_locks[tar], INUSE, AVAILABLE);
    __syncthreads();

    // copy root node elements into shared memory for performing merge sort
    copy_arr1_to_arr2(heap -> arr, ROOT_NODE_IDX * BATCH_SIZE, arr1_shared_mem, 0, BATCH_SIZE);
    memset_arr(heap -> arr, ROOT_NODE_IDX * BATCH_SIZE, INT_MAX, BATCH_SIZE);
    __threadfence();
    // copy partial buffer in arr2_shared mem
    copy_arr1_to_arr2(partial_buffer -> arr, 0, arr2_shared_mem, 0, partial_buffer -> size);
    __syncthreads();

    // merge sort partial buffer in arr2 with root node in arr1
    merge_and_sort(arr1_shared_mem, BATCH_SIZE, arr2_shared_mem, partial_buffer -> size, merged_array_shared_mem);

    // put back to partial buffer since never used
    copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, partial_buffer -> arr, 0, partial_buffer -> size);
    __threadfence();

    // copy back to arr1
    copy_arr1_to_arr2(merged_array_shared_mem, 0, arr1_shared_mem, 0, BATCH_SIZE);

    int left = 0, right = 0, cur = 1;
    int largest_left = 0, largest_right = 0;

    while(1) {
        
        if((cur << 1) >= NUMBER_OF_NODES) {
            // current node is last level of heap so no child 
            break;
        }

        left = cur << 1;
        right = left + 1;

        if(my_thread_id == MASTER_THREAD) {
            // take lock on left and right child with mainatining order to avoid possible deadlock
            take_lock(&heap_locks[left], AVAILABLE, INUSE);
            take_lock(&heap_locks[right], AVAILABLE, INUSE);
        }
        
        __syncthreads();

        copy_arr1_to_arr2(heap -> arr, left * BATCH_SIZE, arr2_shared_mem, 0, BATCH_SIZE);
        memset_arr(heap -> arr, left * BATCH_SIZE, INT_MAX, BATCH_SIZE);
        __threadfence();
        copy_arr1_to_arr2(heap -> arr, right * BATCH_SIZE, arr3_shared_mem, 0, BATCH_SIZE);
        memset_arr(heap -> arr, right * BATCH_SIZE, INT_MAX, BATCH_SIZE);
        __threadfence();

        largest_left = arr2_shared_mem[BATCH_SIZE - 1];
        largest_right = arr3_shared_mem[BATCH_SIZE - 1];

        merge_and_sort(arr2_shared_mem, BATCH_SIZE, arr3_shared_mem, BATCH_SIZE, merged_array_shared_mem);

        // swap left right to avoid code duplication
        if(largest_left > largest_right) {
            int temp = left;
            left = right;
            right = temp;
        }
        
        // now right will be largest element

        copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, heap -> arr, right * BATCH_SIZE, BATCH_SIZE);
        __threadfence();

        // release right chuld lock
        if(my_thread_id == MASTER_THREAD) {
            release_lock(&heap_locks[right], INUSE, AVAILABLE);
        }
        __syncthreads();
        copy_arr1_to_arr2(merged_array_shared_mem, 0, arr2_shared_mem, 0, BATCH_SIZE);
        
        // optimisation for early breaking since current node is already heapifed with this condition
        if(arr1_shared_mem[BATCH_SIZE - 1] <= arr2_shared_mem[0]) {
            copy_arr1_to_arr2(arr2_shared_mem, 0, heap -> arr, left * BATCH_SIZE, BATCH_SIZE);
            __threadfence();
            if(my_thread_id == MASTER_THREAD) {
                release_lock(&heap_locks[left], INUSE, AVAILABLE);
            }
            __syncthreads();
            break;
        }
        // merge sort current node and left node
        merge_and_sort(arr1_shared_mem, BATCH_SIZE, arr2_shared_mem, BATCH_SIZE, merged_array_shared_mem);
        // copy smaaler part in current node
        copy_arr1_to_arr2(merged_array_shared_mem, 0, heap -> arr, cur * BATCH_SIZE, BATCH_SIZE);
        __threadfence();
       
        if(my_thread_id == MASTER_THREAD) {
            release_lock(&heap_locks[cur], INUSE, AVAILABLE);
        }
        __syncthreads();
        copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, arr1_shared_mem, 0, BATCH_SIZE);
        cur = left;
        __syncthreads();
    }
    __syncthreads();
    // copy current array to global heap before releasing lock
    copy_arr1_to_arr2(arr1_shared_mem, 0, heap -> arr, cur * BATCH_SIZE, BATCH_SIZE);
    __threadfence();

    if(my_thread_id == MASTER_THREAD) {
        release_lock(&heap_locks[cur], INUSE, AVAILABLE);
    }
    __syncthreads();

}

__host__ void heap_init() {
    // allocate space for heap, partial buffer and heap locks
    gpuErrchk( cudaMalloc(&d_partial_buffer, sizeof(Partial_Buffer)));
    gpuErrchk( cudaMalloc(&d_heap, sizeof(Heap))); // need to fill with INT_MAX
    gpuErrchk( cudaMalloc((void**)&d_heap_locks, (1 + NUMBER_OF_NODES) * sizeof(int)) );

    for(int i = 0 ; i < NUMBER_OF_CUDA_STREAMS ; i++)
        cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);

    gpuErrchk( cudaMemsetAsync(d_heap_locks, AVAILABLE, (1 + NUMBER_OF_NODES) * sizeof(int), get_current_stream()) );
    next_stream_id();
    heap_init<<<ceil(HEAP_CAPACITY / 1024), 1024, 0, get_current_stream()>>>(d_heap, d_partial_buffer);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

__host__ cudaStream_t get_current_stream() {
    return stream[stream_id];
}

__host__ void next_stream_id() {
    stream_id++;
    if (stream_id >= NUMBER_OF_CUDA_STREAMS) {
        stream_id -= NUMBER_OF_CUDA_STREAMS;
    }
}

__host__ int get_kernel_id() {
    return kernel_id++;
}

__host__ void insert_keys(int *items_to_be_inserted, int total_num_of_keys_insertion) {
    // call insert kernel according to number of elements to be inserted inside heap
    if (total_num_of_keys_insertion < 0) {
        return;
    }

    int num_of_keys_insertion_per_kernel = BATCH_SIZE;
    for(int i = 0 ; i < total_num_of_keys_insertion; i += BATCH_SIZE) {
        num_of_keys_insertion_per_kernel = min(total_num_of_keys_insertion - i, BATCH_SIZE);
        td_insertion<<<1, BLOCK_SIZE, 0, get_current_stream()>>>(items_to_be_inserted + i, num_of_keys_insertion_per_kernel, d_heap_locks, d_partial_buffer, d_heap, get_kernel_id());
        next_stream_id();
    }

}

__host__ void delete_keys(int *items_to_be_deleted) {
    // launch kernel for batch deletion
    td_delete<<<1, BLOCK_SIZE, 0, get_current_stream()>>>(items_to_be_deleted, d_heap_locks, d_partial_buffer, d_heap, get_kernel_id());
    next_stream_id();
}


__host__ void heap_finalise() {
    // deallocates memory allocated for heap
    cudaFree(d_partial_buffer);
    cudaFree(d_heap);
    cudaFree(d_heap_locks);
    for(int i = 0 ; i < NUMBER_OF_CUDA_STREAMS ; i++)
        cudaStreamDestroy(stream[i]);
}
