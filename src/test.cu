#include "td_ins-td_del.h"
#include "sequential_heap.h"
#include <bits/stdc++.h>
#include <ctime>


int arr[HEAP_CAPACITY];
int received_arr[HEAP_CAPACITY];
Heap *b = (Heap*)malloc(sizeof(Heap));

// void test_sort()
// {
//     int n = min(BATCH_SIZE, 16);
//     int h_arr[n];
//     int arr[n];
//     for(int i = 0 ; i < n ; i++)
//         h_arr[i] = arr[i] = rand() % 200;
//     sort(arr, arr + n);
        
//     for(int i = 0 ;i < 16 ;i++)
//         cout << h_arr[i]<<" ";
//     cout<<"\n";
//     int *d_arr;
//     gpuErrchk( cudaMalloc((void**)&d_arr, n * sizeof(int)));
//     gpuErrchk( cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice));
//     td_insertion<<<1, BLOCK_SIZE>>>(d_arr, n, d_heap_locks, d_partial_buffer, d_heap);
// 	cudaDeviceSynchronize();
//     gpuErrchk( cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
//     for(int i = 0 ;i < 16 ;i++)
//         cout << h_arr[i]<<" ";
//     cout << "\n";
//     bool correct = 1;
//     for(int i = 0 ; i < n ; i++)
//         if(arr[i] != h_arr[i])
//         correct = 0;
//     cout << ((correct)?"Success\n":"Failed!\n");

// }

void test_merge()
{
    int n = 0;
    int m = 2;
    int total = n + m;
    int x = 10000;
    int h_arr1[n], h_arr2[m], h_arr3[n + m];
    int verify[n + m];
    int c = 0;
    for(int i = 0 ; i < n ; i++) {
        h_arr1[i] = rand() % x;
    }
    

    for(int i = 0 ; i < m ; i++) {
        h_arr2[i] = rand() % x;
    }
    sort(h_arr2, h_arr2 + m);
    sort(h_arr1, h_arr1 + n);
    for(int i = 0 ; i < n ; i++) {
        verify[c++] = h_arr1[i];
    }
    

    for(int i = 0 ; i < m ; i++) {
        verify[c++] = h_arr2[i];
    }

    int *d_arr1, *d_arr2, *d_arr3;
    gpuErrchk( cudaMalloc((void**)&d_arr1, n * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr1, h_arr1 , n * sizeof(int), cudaMemcpyHostToDevice)); 
    gpuErrchk( cudaMalloc((void**)&d_arr2, m * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr2, h_arr2 , m * sizeof(int), cudaMemcpyHostToDevice)); 
    gpuErrchk( cudaMalloc((void**)&d_arr3, total * sizeof(int)));

    int num_thread_per_block = 1024;
    int number_of_blocks = (total / num_thread_per_block) + 1;
    merge_and_sort_cpu_test<<<number_of_blocks, num_thread_per_block>>>(d_arr1, n, d_arr2, m, d_arr3);
    cudaDeviceSynchronize();
    gpuErrchk( cudaMemcpy(h_arr3, d_arr3, total * sizeof(int), cudaMemcpyDeviceToHost));
    
    // for(int i = 0 ; i < total ; i++) {
    //     cout << verify[i] << " ";
    // }
    // cout << "\n";
    sort(verify, verify + total);
    bool correct = 1;
    for(int i = 0; i < total ; i++)
        if(h_arr3[i] != verify[i]) {
            correct = 0;
            // cout << i << " " <<h_arr3[i] << " " << verify[i] << "\n";
            // break;
        }

    cout << ((correct)?"Success\n":"Failed!\n");

}

void test_insertion()
{
    int n = 60000;
    n = NUMBER_OF_NODES - 1;
    cout << n<<"\n";
    for(int i = 0 ; i < HEAP_CAPACITY; i++)
        arr[i] = rand() % 100000 + 10;
    // for(int i = 1 ; i < n; i++)
    //     cout << arr[i] << " ";
    //     cout << "\n";
    long double total_time = 0;
    int number_of_streams = 20;
    cudaStream_t stream[number_of_streams];
    for(int i = 1 ; i < number_of_streams ; i++)
        cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
    int *d_arr;
    gpuErrchk( cudaMalloc((void**)&d_arr, HEAP_CAPACITY * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr, (arr) , HEAP_CAPACITY * sizeof(int), cudaMemcpyHostToDevice)); 
    cudaDeviceSynchronize();
    std::clock_t c_start = std::clock();
    for(int i = 1; i <n  ; i++)
    {
        // cudaDeviceSynchronize();
        td_insertion<<<1, BLOCK_SIZE,0, stream[i%(number_of_streams - 1) + 1]>>>(d_arr + i*BATCH_SIZE, BATCH_SIZE, d_heap_locks, d_partial_buffer, d_heap, i);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    std::clock_t c_end = std::clock();
    long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "GPU time used: " << time_elapsed_ms << " ms\n";

    priority_queue<int> pq;
    c_start = std::clock();
    for(int i = 1; i < n ; i++)
    {
        for(int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE ; j++)
        {
            pq.push(arr[j]);
        }
    }
    // while(pq.size()!=0) {
    //     pq.pop();
    // }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU-STL time used: " << time_elapsed_ms << " ms\n";

    CPU_Heap my_heap(HEAP_CAPACITY);
    c_start = std::clock();
    int c = 0;
    for(int i = 1; i < n ; i++)
    {
        for(int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE ; j++)
        {
            c++;
            my_heap.push(arr[j]);
        }
    }
    // while(pq.size()!=0) {
    //     pq.pop();
    // }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU my heap time used: " << time_elapsed_ms << " ms\n";

    gpuErrchk( cudaMemcpy(b, d_heap, sizeof(Heap), cudaMemcpyDeviceToHost));
    sort(arr + BATCH_SIZE, arr + n*BATCH_SIZE);
    // for(int i = 1 ; i < n ; i++)
    //     cout << arr[i] << " " << b ->arr[i] << "\n";
    bool correct = 1;
    for(int i = BATCH_SIZE ; i < 2*BATCH_SIZE ; i++)
        if (arr[i] != b->arr[i] or my_heap.pop() != arr[i]){
            cout << arr[i] << " " << b -> arr[i] << " ";
            correct = 0;
            // break;
        }
    
    for(int i = 1 ; i < n; i++)
    {
        int child_low = i * BATCH_SIZE;
        int child_high = child_low + BATCH_SIZE - 1;
        int par = i / 2;
        int par_low = par*BATCH_SIZE;
        int par_high = par_low + BATCH_SIZE - 1;
        for(int j = child_low + 1 ; j <= child_high ; j++)
            if(b -> arr[j - 1] > b->arr[j])
                correct = 0;
        if(par != 0) {
            if(b->arr[child_low] < b-> arr[par_high])
            {
                cout <<par  << " "<< b->arr[child_low] << " " << b-> arr[par_high] << "\n";
                correct = 0;
            }
            // cout << b->arr[child_low] << "\n";
        }
        
    }
    sort(b->arr + BATCH_SIZE, b->arr + HEAP_CAPACITY);
    for(int i = BATCH_SIZE ; i < n * BATCH_SIZE ; i++)
        if(b->arr[i] != arr[i]) {
            cout << i << " "<< arr[i] << " " << b -> arr[i] << "\n";
            correct=0;
        }


    cout << ((correct)?"Success\n":"Failed!\n");
}

void test_deletion() {

    int n = 60000;
    n = NUMBER_OF_NODES - 1;
    cout<<n<<"\n";
    for(int i = 0 ; i < HEAP_CAPACITY; i++)
        arr[i] = rand() % 5000000+ 10;
    // for(int i = 1 ; i < n ; i++)
    // cout << arr[i] << " ";cout << "\n";
    long double total_time = 0;
    int number_of_streams = 30;
    cudaStream_t stream[number_of_streams];
    for(int i = 1 ; i < number_of_streams ; i++)
        cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
    int *d_arr;
    gpuErrchk( cudaMalloc((void**)&d_arr, HEAP_CAPACITY * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr, (arr) , HEAP_CAPACITY * sizeof(int), cudaMemcpyHostToDevice)); 

    int *d_arr_rec;
    gpuErrchk( cudaMalloc((void**)&d_arr_rec, HEAP_CAPACITY * sizeof(int)));
    cudaDeviceSynchronize();
    int c = 1;
    std::clock_t c_start = std::clock();
    for(int i = 1; i < n  ; i++)
    {
        td_insertion<<<1, BLOCK_SIZE,0, stream[i%(number_of_streams - 1) + 1]>>>(d_arr + i*BATCH_SIZE, BATCH_SIZE, d_heap_locks, d_partial_buffer, d_heap, c++);
    }
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    cudaDeviceSynchronize();
    for(int i = 1 ; i < n; i++) {
        td_delete<<<1, BLOCK_SIZE,0, stream[i%(number_of_streams - 1) + 1]>>>(d_arr_rec + i*BATCH_SIZE, d_heap_locks, d_partial_buffer, d_heap, c++);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );
    }

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    std::clock_t c_end = std::clock();
    long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "GPU time used: " << time_elapsed_ms << " ms\n";

    priority_queue<int> pq;
    c_start = std::clock();
    for(int i = 1; i < n ; i++)
    {
        for(int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE ; j++)
        {
            pq.push(arr[j]);
        }
    }
    // while(pq.size()!=0) {
    //     pq.pop();
    // }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU-STL time used: " << time_elapsed_ms << " ms\n";

    CPU_Heap my_heap(HEAP_CAPACITY);
    c_start = std::clock();
    for(int i = 1; i < n ; i++)
    {
        for(int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE ; j++)
        {
            my_heap.push(arr[j]);
        }
    }
    // while(not my_heap.is_empty()) {
    //     my_heap.pop();
    // }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU my heap time used: " << time_elapsed_ms << " ms\n";

    gpuErrchk( cudaMemcpy(received_arr, d_arr_rec, HEAP_CAPACITY * sizeof(int), cudaMemcpyDeviceToHost));
    sort(arr + BATCH_SIZE, arr + n * BATCH_SIZE);
    // for(int i = 1 ; i < 4 ; i++)
    //     cout << received_arr[i] << " " << b ->arr[i] << "\n";
    bool correct = 1;
    for(int i = BATCH_SIZE ; i < n*BATCH_SIZE ; i++) {
        if (arr[i] != received_arr[i]) {
            correct = 0;
            cout << arr[i] << " " << received_arr[i] << " " << i << "\n";
            // break;
        }
    }

    cout << ((correct)?"Success\n":"Failed!\n");
}

int main()
{
    heap_init();
    // srand(0);
    srand(time(NULL));
 
    // test_insertion();
    test_deletion();
    // test_merge();
    
}