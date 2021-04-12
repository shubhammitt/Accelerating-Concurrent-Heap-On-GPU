#include<iostream>
using namespace std;

// BLOCK_SIZE >= BATCH_SIZE always to ensure maximum intra-node parallism and correctness of code
// For full usage of GPU resources, BLOCK_SIZE = BATCH_SIZE
#define BATCH_SIZE 1024
#define BLOCK_SIZE 1024
// Following code is from https://stackoverflow.com/a/14038590 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Algorithm referenced from https://wiki.rice.edu/confluence/download/attachments/4435861/comp322-s12-lec28-slides-JMC.pdf?version=1&modificationDate=1333163955158
__device__ void bitonic_sort(int *arr, int size)
{
    // assuming size = power of 2
    int my_thread_id = threadIdx.x;
    if (my_thread_id < size)
    {
        int maximum = 0, minimum = 0, other_idx = 0, i = 2, j = 2;;
        int my_batch_number = my_thread_id >> 1; // parity of batch number will tell which operation to perform in that batch: min/max for even/odd respectively
        for (i = 2; i <= size ; i <<= 1, my_batch_number >>= 1)
        {
            for(j = i; j >= 2 ; j >>= 1)
            {
                int steps_to_look_ahead = j >> 1;
                if (my_thread_id % j < steps_to_look_ahead) // only first half of any batch can be active
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
}

__global__ void td_insertion(int *items_to_be_inserted, int number_of_items_to_be_inserted)
{
    __shared__ int items_to_be_inserted_shared_mem[BATCH_SIZE];
    int my_thread_id = threadIdx.x;

    if (my_thread_id < BATCH_SIZE)
    {
        items_to_be_inserted_shared_mem[my_thread_id] = items_to_be_inserted[my_thread_id];
    }
    __syncthreads();
    bitonic_sort(items_to_be_inserted_shared_mem, number_of_items_to_be_inserted);
    
    if (my_thread_id < BATCH_SIZE)
    {
        items_to_be_inserted[my_thread_id] = items_to_be_inserted_shared_mem[my_thread_id];
    }

}

int main()
{
    srand(time(NULL));
    int n = 1024;
    int h_arr[n];
    for(int i = 0 ; i < n ; i++)
        h_arr[i] = rand() % 10;
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    td_insertion<<<1, BLOCK_SIZE>>>(d_arr, n);
	cudaDeviceSynchronize();
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    bool correct = true;
    for(int i = 1 ; i < n ; i++)
        if(h_arr[i] < h_arr[i-1])
            correct = 0;
    cout << correct <<"\n";

}