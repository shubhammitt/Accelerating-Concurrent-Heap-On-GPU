#include "td_ins-td_del-runtime.h"
#include "sequential_heap.h"
#include <bits/stdc++.h>
#include <ctime>
#include<assert.h>
#include <unistd.h>
using namespace std;

int *arr;
int *received_arr;

void test() {
    int n = NUMBER_OF_NODES;
    int heap_capacity = n * BATCH_SIZE;
    arr = new int[heap_capacity];
    received_arr = new int[heap_capacity];

    // initialise array for input
    for(int i = 0 ; i < heap_capacity; i++)
        arr[i] = rand() % 5000000;
    
    // initialise input array for device
    int *d_arr;
    gpuErrchk( cudaMalloc((void**)&d_arr, heap_capacity * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr, (arr) , heap_capacity * sizeof(int), cudaMemcpyHostToDevice)); 

    int *d_arr_rec;
    gpuErrchk( cudaMalloc((void**)&d_arr_rec, heap_capacity * sizeof(int)));
    cudaDeviceSynchronize();
    std::clock_t c_start = std::clock();

    for(int i = 0; i < n  ; i++) {
        insert_keys(d_arr + i * BATCH_SIZE, BATCH_SIZE);   
    }
    for(int i = 0 ; i < n; i++) {
        delete_keys(d_arr_rec + i * BATCH_SIZE);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    std::clock_t c_end = std::clock();
    long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "GPU time used: " << time_elapsed_ms << " ms\n";

    priority_queue<int> pq;
    c_start = std::clock();
    // for(int i = 1; i < n ; i++)
    // {
    //     for(int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE ; j++)
    //     {
    //         pq.push(arr[j]);
    //     }
    // }
    // while(pq.size()!=0) {
    //     pq.pop();
    // }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU-STL time used: " << time_elapsed_ms << " ms\n";

    // CPU_Heap my_heap(heap_capacity);
    c_start = std::clock();
    // for(int i = 1; i < n ; i++)
    // {
    //     for(int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE ; j++)
    //     {
    //         my_heap.push(arr[j]);
    //     }
    // }
    // while(not my_heap.is_empty()) {
    //     my_heap.pop();
    // }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU my heap time used: " << time_elapsed_ms << " ms\n";

    gpuErrchk( cudaMemcpy(received_arr, d_arr_rec, heap_capacity * sizeof(int), cudaMemcpyDeviceToHost));

    // verify
    sort(arr, arr + heap_capacity);
    bool correct = 1;
    for(int i = 0 ; i < heap_capacity ; i++) {
        if (arr[i] != received_arr[i]) {
            correct = 0;
            break;
        }
    }

    cout << ((correct)?"Success\n":"Failed!\n");
}

int main()
{
    heap_init();
    // srand(0);
    srand(time(NULL));
    test();
    
}