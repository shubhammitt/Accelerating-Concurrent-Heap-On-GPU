/*
 * Author: Shubham Mittal
 * IIITD, 2018101
 * GPU Project: Accelerating Concurrent Heap on GPUs
*/
#include "td_ins-td_del.h"


__device__ int get_lock_state(int node_idx, int *heap_locks) {
    return heap_locks[node_idx];
}


__device__ void take_lock(int *lock, int lock_state_1, int lock_state_2) {
    while (atomicCAS(lock, lock_state_1, lock_state_2) != lock_state_1);
}


__device__ int try_lock(int *lock, int lock_state_1, int lock_state_2) {
    return atomicCAS(lock, lock_state_1, lock_state_2);
}


__device__ void release_lock(int *lock, int lock_state_1, int lock_state_2) {
    atomicCAS(lock, lock_state_1, lock_state_2);
}


__global__ void heap_init(Heap *heap, Partial_Buffer *partial_buffer) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    heap -> global_id = 1;
    heap -> global_idx = 1;
    heap -> size = 0;
    partial_buffer -> size = 0;
    if (index < HEAP_CAPACITY) {
        heap -> arr[index] = INT_MAX;
    }
    if (index < PARTIAL_BUFFER_CAPACITY) {
        partial_buffer -> arr[threadIdx.x] = INT_MAX;   
    }
}


__device__ int bit_reversal(int n, int level) {
    if (n <= 3)
        return n;

    int ans = 1 << (level--);
    while(n != 1) {
        ans += (n & 1) << (level--);
        n >>= 1;
    }
    return ans;

}
__device__ void copy_arr1_to_arr2(int *arr1, int from_arr1_idx1, int to_arr1_idx2, int *arr2, int from_arr2_idx1) {
    int my_thread_id = threadIdx.x;
    int n = to_arr1_idx2 - from_arr1_idx1;
    if (my_thread_id < n) {
        arr2[from_arr2_idx1 + my_thread_id] = arr1[from_arr1_idx1 + my_thread_id];
    }
}


__device__ void memset_arr(int *arr, int from_arr_idx1, int to_arr_idx2, int val) {
    int my_thread_id = threadIdx.x;
    int n = to_arr_idx2 - from_arr_idx1;
    if (my_thread_id < n) {
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
    int low = 0, mid = 0;
    int ans = high;
    while (low <= high)
    {
        mid = (low + high) >> 1;
        if (arr1[mid] >= search and consider_equality) {
            ans = mid;
            high = mid - 1;
        }
        else if (arr1[mid] > search) {
            ans = mid;
            high = mid - 1;
        }
        else
            low = mid + 1;
    }
    return ans;
}


__device__ void merge_and_sort(int *arr1, int idx1, int *arr2, int idx2, int *merged_arr) {
    int my_thread_id = threadIdx.x;

    if (my_thread_id < idx1) {
        merged_arr[my_thread_id + binary_search(arr2, idx2, arr1[my_thread_id], 1)] = arr1[my_thread_id];
    }

    if (my_thread_id < idx2) {
        merged_arr[my_thread_id + binary_search(arr1, idx1, arr2[my_thread_id], 0)] = arr2[my_thread_id];
    }

    __syncthreads();
    
}


__global__ void td_insertion(int *items_to_be_inserted, int number_of_items_to_be_inserted, int *heap_locks, Partial_Buffer *partial_buffer, Heap *heap, int my_id) {
    /*
     * number_of_items_to_be_inserted <= BATCH_SIZE
    */
    const int double_batch_size = BATCH_SIZE << 1;
    __shared__ int items_to_be_inserted_shared_mem[BATCH_SIZE];
    __shared__ int array_to_be_merged_shared_mem[BATCH_SIZE];
    __shared__ int merged_array_shared_mem[double_batch_size];

    int my_thread_id = threadIdx.x;

    // memset_arr(items_to_be_inserted_shared_mem, number_of_items_to_be_inserted, BATCH_SIZE);
    // memset_arr(array_to_be_merged_shared_mem, 0, BATCH_SIZE);
    // memset_arr(merged_array_shared_mem, 0, BATCH_SIZE << 1);

    // copy keys to be inserted in shared memory
    copy_arr1_to_arr2(items_to_be_inserted, 0, number_of_items_to_be_inserted, items_to_be_inserted_shared_mem, 0);
    __syncthreads();

    // sort the keys to be inserted
    bitonic_sort(items_to_be_inserted_shared_mem, number_of_items_to_be_inserted);

    // take root node lock
    if (my_thread_id == MASTER_THREAD){
        while(atomicCAS(&(heap -> global_id), my_id, 0) != my_id);
        take_lock(&heap_locks[ROOT_NODE_IDX], AVAILABLE, INUSE);
        heap -> global_id = my_id + 1;
    }

    __syncthreads();

    // copy partial buffer into shared memory
    copy_arr1_to_arr2(partial_buffer -> arr, 0, partial_buffer -> size, array_to_be_merged_shared_mem, 0);
    __syncthreads();

    // merge partial buffer and keys to be inserted
    merge_and_sort(items_to_be_inserted_shared_mem, number_of_items_to_be_inserted, \
            array_to_be_merged_shared_mem, partial_buffer -> size, merged_array_shared_mem);
    
    int combined_size = partial_buffer -> size + number_of_items_to_be_inserted;
    if (combined_size >= BATCH_SIZE) {

        // copy batch_size into insertion key list
        copy_arr1_to_arr2(merged_array_shared_mem, 0, BATCH_SIZE, items_to_be_inserted_shared_mem, 0);

        // copy rest over in partial buffer and update its size
        copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, combined_size - BATCH_SIZE, partial_buffer -> arr, 0);
        __syncthreads();

        // update partial buffer size
        if (my_thread_id == MASTER_THREAD)
            partial_buffer -> size = combined_size - BATCH_SIZE;
    }
    else {
        if (heap -> size == 0) {
            // transfer all keys in partial buffer
            copy_arr1_to_arr2(merged_array_shared_mem, 0, combined_size, partial_buffer -> arr, 0);
            if (my_thread_id == MASTER_THREAD)
                partial_buffer -> size = combined_size;
        }
        else {
            // copy partial buffer into shared array
            copy_arr1_to_arr2(merged_array_shared_mem, 0, combined_size, array_to_be_merged_shared_mem, 0);
            
            // copy root node into shared memory
            copy_arr1_to_arr2(heap -> arr, ROOT_NODE_IDX * BATCH_SIZE, ROOT_NODE_IDX * BATCH_SIZE + BATCH_SIZE, items_to_be_inserted_shared_mem, 0);
            __syncthreads();

            // merge partial buffer with root node
            merge_and_sort(items_to_be_inserted_shared_mem, BATCH_SIZE, array_to_be_merged_shared_mem, combined_size, merged_array_shared_mem);
        
            // copy back to root node
            copy_arr1_to_arr2(merged_array_shared_mem, 0, BATCH_SIZE, heap -> arr, ROOT_NODE_IDX * BATCH_SIZE);

            // copy to partial buffer
            copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, BATCH_SIZE + combined_size, partial_buffer -> arr, 0);

            if (my_thread_id == MASTER_THREAD)
                partial_buffer -> size = combined_size;
        }
        if (my_thread_id == MASTER_THREAD)
            release_lock(&heap_locks[ROOT_NODE_IDX], INUSE, AVAILABLE);
        return;
    }

    if (my_thread_id == MASTER_THREAD)
        (heap -> size += 1);
    __syncthreads();

    int tar = heap -> size, level = __log2f(tar); // may have floating point error, default to -1
    // int dummy_tar = tar;
    // while(dummy_tar) {
    //     level++;
    //     dummy_tar >>= 1;
    // }

    tar = bit_reversal(tar, level);
    
    // take lock on target node 
    if (tar != ROOT_NODE_IDX) {
        if (my_thread_id == MASTER_THREAD) {
            take_lock(&heap_locks[tar], AVAILABLE, INUSE);
        }
        __syncthreads();
    }

    int low = 0, high = 0, cur = ROOT_NODE_IDX;;
    while (cur != tar) {
        if (get_lock_state(tar, heap_locks) == MARKED) { // next delete operation can cooperate with current insert operation
            break;
        }
        
        low = cur * BATCH_SIZE;
        high = low + BATCH_SIZE;
        // copy current node to shared mem
        copy_arr1_to_arr2(heap -> arr, low, high, array_to_be_merged_shared_mem, 0);
        __syncthreads();

        // merger current batch with insertion list in shared mem
        merge_and_sort(array_to_be_merged_shared_mem, BATCH_SIZE, items_to_be_inserted_shared_mem, BATCH_SIZE, merged_array_shared_mem);

        // copy back to current batch
        copy_arr1_to_arr2(merged_array_shared_mem, 0, BATCH_SIZE, heap -> arr, low);

        __syncthreads();
        // copy to insertion list
        copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, double_batch_size, items_to_be_inserted_shared_mem, 0);

        cur = tar >> (--level);

        if(my_thread_id == MASTER_THREAD) {
            if (cur != tar) {
                take_lock(&heap_locks[cur], AVAILABLE, INUSE);
            }
            release_lock(&heap_locks[cur >> 1], INUSE, AVAILABLE);
        }
        __syncthreads();
    }

    if(my_thread_id == MASTER_THREAD) {
        try_lock(&heap_locks[tar], TARGET, INUSE);
    }
    __syncthreads();
    tar = (get_lock_state(tar, heap_locks) == INUSE) ? tar : 1;
    copy_arr1_to_arr2(items_to_be_inserted_shared_mem, 0, BATCH_SIZE, heap -> arr , tar * BATCH_SIZE);

    if(my_thread_id == MASTER_THREAD) {
        if(tar != cur) {
            release_lock(&heap_locks[tar], get_lock_state(tar, heap_locks), AVAILABLE);
        }
        release_lock(&heap_locks[cur], INUSE, AVAILABLE);
    }
}

__global__ void td_delete(int *items_deleted, int *heap_locks, Partial_Buffer *partial_buffer, Heap *heap, int my_id) {
    
    int my_thread_id = threadIdx.x;
    const int double_batch_size = BATCH_SIZE << 1;
    __shared__ int arr1_shared_mem[BATCH_SIZE];
    __shared__ int arr2_shared_mem[BATCH_SIZE];
    __shared__ int arr3_shared_mem[BATCH_SIZE];
    __shared__ int merged_array_shared_mem[double_batch_size];


    // take root node lock
    if (my_thread_id == MASTER_THREAD)
    {
        while(atomicCAS(&(heap -> global_id), my_id, 0) != my_id);
        take_lock(&heap_locks[ROOT_NODE_IDX], AVAILABLE, INUSE);
        heap -> global_id = my_id + 1;
        // printf("%d \n", my_id);
    }
    __syncthreads();

    // heap is empty
    if (heap -> size == 0) {
        if (partial_buffer -> size != 0) {
            copy_arr1_to_arr2(partial_buffer -> arr, 0, partial_buffer -> size, items_deleted + (heap -> global_idx) * BATCH_SIZE, 0);
            __syncthreads();
            if(my_thread_id == MASTER_THREAD) {
                partial_buffer -> size = 0;
                heap -> global_idx += 1;
            }
            __syncthreads();
        }
        if (my_thread_id == MASTER_THREAD) {
            release_lock(&heap_locks[ROOT_NODE_IDX], INUSE, AVAILABLE);
        }
        __syncthreads();
        return;
    }

    // copy root into shared mem arr1 to be used now and later too
    copy_arr1_to_arr2(heap -> arr, ROOT_NODE_IDX * BATCH_SIZE, ROOT_NODE_IDX * BATCH_SIZE + BATCH_SIZE, arr1_shared_mem, 0);
    // copy root node into list of deleted mem
    copy_arr1_to_arr2(arr1_shared_mem, 0, BATCH_SIZE, items_deleted, heap -> global_idx * BATCH_SIZE);
    __syncthreads();
    if(my_thread_id == MASTER_THREAD) {
        heap -> global_idx += 1;
    }
    __syncthreads();
    int tar = heap -> size;

    if (tar == 1) { // WARNING: not written in pseudocode
        if (partial_buffer -> size == 0) {
            if (my_thread_id == MASTER_THREAD)
                heap -> size = 0;
        }
        else {
            copy_arr1_to_arr2(partial_buffer -> arr, 0 , partial_buffer -> size, heap -> arr, ROOT_NODE_IDX * BATCH_SIZE);
            __syncthreads();
            partial_buffer -> size = 0;
        }
        if(my_thread_id == MASTER_THREAD) {
            release_lock(&heap_locks[ROOT_NODE_IDX], INUSE, AVAILABLE);
        }
        __syncthreads();
        return; 
    }

    int level = __log2f(tar);
    tar = bit_reversal(tar, level);
    int cur = 1;
    
    if (my_thread_id == MASTER_THREAD) {
        try_lock(&heap_locks[tar], TARGET, MARKED);
    } 

    __syncthreads(); // necessary so that master thread do not decrement while other threads are initialising tar
    if (my_thread_id == MASTER_THREAD)
        heap -> size -= 1;
    

    if (get_lock_state(tar, heap_locks) == MARKED) {
        while(get_lock_state(tar, heap_locks) != AVAILABLE);
    }
    else {
        if (my_thread_id == MASTER_THREAD) {
            take_lock(&heap_locks[tar], AVAILABLE, INUSE);
        }
        __syncthreads();
        // root node elements are already copied in arr1
        copy_arr1_to_arr2(heap -> arr, tar * BATCH_SIZE, (tar + 1) * BATCH_SIZE, arr1_shared_mem, 0);
        __syncthreads();
        memset_arr(heap -> arr, tar * BATCH_SIZE, (tar + 1) * BATCH_SIZE, INT_MAX);
        __syncthreads();

        if (my_thread_id == MASTER_THREAD) {
            release_lock(&heap_locks[tar], INUSE, AVAILABLE);
        }        
    }

    // copy partial buffer in arr2_shared mem
    copy_arr1_to_arr2(partial_buffer -> arr, 0, partial_buffer -> size, arr2_shared_mem, 0);
    __syncthreads();

    // merge sort partial buffer in arr2 with root node in arr1
    merge_and_sort(arr1_shared_mem, BATCH_SIZE, arr2_shared_mem, partial_buffer -> size, merged_array_shared_mem);

    // put back to partial buffer since never used
    copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, BATCH_SIZE + partial_buffer -> size, partial_buffer -> arr, 0);

    // copy back to arr1
    copy_arr1_to_arr2(merged_array_shared_mem, 0, BATCH_SIZE, arr1_shared_mem, 0);

    int left = 0, right = 0;
    int largest_left = 0, largest_right = 0;
    while(1) {
        
        if((cur << 1) >= NUMBER_OF_NODES) {
            break; // same code after while loop
        }

        left = cur << 1;
        right = left + 1;

        if(my_thread_id == MASTER_THREAD) {
            // take lock on left and right child with mainatining order to avoid possible deadlock
            take_lock(&heap_locks[left], AVAILABLE, INUSE);
            take_lock(&heap_locks[right], AVAILABLE, INUSE);
        }
        __syncthreads();


        copy_arr1_to_arr2(heap -> arr, left * BATCH_SIZE, (left * BATCH_SIZE) + BATCH_SIZE, arr2_shared_mem, 0);
        copy_arr1_to_arr2(heap -> arr, right * BATCH_SIZE, (right * BATCH_SIZE) + BATCH_SIZE, arr3_shared_mem, 0);
        __syncthreads();

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

        copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, double_batch_size, heap -> arr, right * BATCH_SIZE);
        __syncthreads();
        if(my_thread_id == MASTER_THREAD) {
            release_lock(&heap_locks[right], INUSE, AVAILABLE);
        }
        copy_arr1_to_arr2(merged_array_shared_mem, 0, BATCH_SIZE, arr2_shared_mem, 0);
        __syncthreads();
        
        // temporary debug comment
        if(arr1_shared_mem[BATCH_SIZE - 1] <= arr2_shared_mem[0]) {
            copy_arr1_to_arr2(arr2_shared_mem, 0, BATCH_SIZE, heap -> arr, left * BATCH_SIZE);
            __syncthreads();
            if(my_thread_id == MASTER_THREAD) {
                release_lock(&heap_locks[left], INUSE, AVAILABLE);
            }
            break;
        }
        merge_and_sort(arr1_shared_mem, BATCH_SIZE, arr2_shared_mem, BATCH_SIZE, merged_array_shared_mem);
        
        copy_arr1_to_arr2(merged_array_shared_mem, 0, BATCH_SIZE, heap -> arr, cur * BATCH_SIZE);
        __syncthreads();

       
        if(my_thread_id == MASTER_THREAD) {
            release_lock(&heap_locks[cur], INUSE, AVAILABLE);
        }
        copy_arr1_to_arr2(merged_array_shared_mem, BATCH_SIZE, double_batch_size, arr1_shared_mem, 0);
        cur = left;
        __syncthreads();
    }

    // copy current array to global heap before releasing lock
    copy_arr1_to_arr2(arr1_shared_mem, 0, BATCH_SIZE, heap -> arr, cur * BATCH_SIZE);
    __syncthreads();

    if(my_thread_id == MASTER_THREAD) {
        release_lock(&heap_locks[cur], INUSE, AVAILABLE);
    }

}
__host__ void heap_init() {
    gpuErrchk( cudaMalloc(&d_partial_buffer, sizeof(Partial_Buffer)));
    gpuErrchk( cudaMalloc(&d_heap, sizeof(Heap))); // need to fill with INT_MAX

    heap_init<<<ceil(HEAP_CAPACITY / 1024.0), 1024>>>(d_heap, d_partial_buffer);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMalloc((void**)&d_heap_locks, NUMBER_OF_NODES * sizeof(int)) );
    gpuErrchk( cudaMemset(d_heap_locks, AVAILABLE, NUMBER_OF_NODES * sizeof(int)) );
}
