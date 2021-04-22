#include "td_ins-td_del-runtime.h"
#include "sequential_heap.h"
#include <bits/stdc++.h>
#include <ctime>
#include<assert.h>
#include <unistd.h>
using namespace std;

int *arr; // input
int *received_arr; // output
int *seq_out;

void input1_ascending(int *arr, int n) {
    for(int i = 0 ; i < n ; i++)
        arr[i] = i;
}


void input2_decsending(int *arr, int n) {
    int m = 1e8;
    for(int i = 0 ; i < n ; i++)
        arr[i] = m--;
}

void input3_random(int *arr, int n) {
    int m = 1e8;
    for(int i = 0 ; i < n ; i++)
        arr[i] = rand() % m;
}

void verify_results(int *h_arr, int *d_arr, int n) {
    bool correct = 1;
    for(int i = 0 ; i < n ; i++) {
        if (h_arr[i] != d_arr[i]) {
            correct = 0;
            cout << h_arr[i] << " " << d_arr[i] << " " << i << "\n";
            break;
        }
    }
    cout << ((correct)?"Success\n":"Failed!\n");
}

void test(int tc) {
    int n = NUMBER_OF_NODES;
    int heap_capacity = n * BATCH_SIZE;

    // alocate input array for host
    arr = new int[heap_capacity];
    seq_out = new int[heap_capacity];
    received_arr = new int[heap_capacity];


    // initialise array for input host
    if(tc == 1)
        input1_ascending(arr, heap_capacity);
    else if(tc == 2)
        input2_decsending(arr, heap_capacity);
    else
        input3_random(arr, heap_capacity);

    
    std::clock_t c_start_gpu = std::clock();
    // initialise data structure
    heap_init();
    // allocate initialise input array for device
    int *d_arr;
    gpuErrchk( cudaMalloc((void**)&d_arr, heap_capacity * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr, arr , heap_capacity * sizeof(int), cudaMemcpyHostToDevice)); 

    // output received array of device
    int *d_arr_rec;
    gpuErrchk( cudaMalloc((void**)&d_arr_rec, heap_capacity * sizeof(int)));

    // start timer
    std::clock_t c_start = std::clock();

    // start insertion
    for(int i = 0; i < n  ; i++) {
        insert_keys(d_arr + i * BATCH_SIZE, BATCH_SIZE);   
    }
    // start deletion
    for(int i = 0 ; i < n; i++) {
        delete_keys(d_arr_rec + i * BATCH_SIZE);
    }

    // wait for kernels to execute
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // end timer
    std::clock_t c_end = std::clock();
    long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "GPU kernel Execution time : \t" << time_elapsed_ms << " ms\n";

    // copy output array from device to host
    gpuErrchk( cudaMemcpy(received_arr, d_arr_rec, heap_capacity * sizeof(int), cudaMemcpyDeviceToHost));

    // free input array
    cudaFree(d_arr);
    // free output array
    cudaFree(d_arr_rec);
    // free heap on device
    heap_finalise();
    std::clock_t c_end_gpu = std::clock();
    time_elapsed_ms = 1000.0 * (c_end_gpu-c_start_gpu) / CLOCKS_PER_SEC;
    std::cout << "GPU Full Execution time :\t" << time_elapsed_ms << " ms\n";


    // STL heap testing
    priority_queue<int> pq;
    c_start = std::clock();
    for(int i = 0; i < n ; i++)
    {
        for(int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE ; j++)
        {
            pq.push(arr[j]);
        }
    }
    while(pq.size()!=0) {
        pq.pop();
    }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU-STL Execution time :\t" << time_elapsed_ms << " ms\n";

    // Sequential Heap testing
    CPU_Heap my_heap(heap_capacity);
    c_start = std::clock();
    for(int i = 0; i < n ; i++)
    {
        for(int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE ; j++)
        {
            my_heap.push(arr[j]);
        }
    }
    int x = 0;
    while(not my_heap.is_empty()) {
        seq_out[x++] = my_heap.pop();
    }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU my heap Execution time:\t" << time_elapsed_ms << " ms\n";

    // verify
    sort(arr, arr + heap_capacity);
    verify_results(arr, received_arr, heap_capacity);
    verify_results(arr, seq_out, heap_capacity);

}

int main()
{
    test(3);
}
