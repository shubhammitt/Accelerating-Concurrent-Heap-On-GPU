#include "td_ins-td_del.h"
#include "sequential_heap.h"
#include <bits/stdc++.h>
#include <ctime>
void test_sort()
{
    int n = min(BATCH_SIZE, 16);
    int h_arr[n];
    int arr[n];
    for(int i = 0 ; i < n ; i++)
        h_arr[i] = arr[i] = rand() % 200;
    sort(arr, arr + n);
        
    for(int i = 0 ;i < 16 ;i++)
        cout << h_arr[i]<<" ";
    cout<<"\n";
    int *d_arr;
    gpuErrchk( cudaMalloc((void**)&d_arr, n * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice));
    td_insertion<<<1, BLOCK_SIZE>>>(d_arr, n, d_heap_locks, d_partial_buffer, d_heap);
	cudaDeviceSynchronize();
    gpuErrchk( cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
    for(int i = 0 ;i < 16 ;i++)
        cout << h_arr[i]<<" ";
    cout << "\n";
    bool correct = 1;
    for(int i = 0 ; i < n ; i++)
        if(arr[i] != h_arr[i])
        correct = 0;
    cout << ((correct)?"Success\n":"Failed!\n");

}

void test_merge()
{
    //  batchsize = 512 for this to work
    int n = min(BATCH_SIZE, 128);
    int h_arr[2*n];
    int h_arr2[2*n];
    int arr[2*n];
    for(int i = 0 ; i < n ; i++)
    {
        h_arr[i] = arr[i] = rand() % 10;
        h_arr2[i] = arr[i + n] = rand() % 10;

    }
    for(int i = 0 ;i < n ;i++)
        cout << h_arr[i] << " " << h_arr2[i]<<"\n";
    cout << "\n";
    n = 2*n;
    sort(arr, arr + n);
    
    int *d_arr; int *d_arr2;
    gpuErrchk( cudaMalloc((void**)&d_arr, n * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMalloc((void**)&d_arr2, n * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr2, h_arr2, n * sizeof(int), cudaMemcpyHostToDevice));
    td_insertion<<<1, BLOCK_SIZE>>>(d_arr, n >> 1, d_heap_locks, d_partial_buffer, d_heap);
    td_insertion<<<1, BLOCK_SIZE>>>(d_arr2, n >> 1, d_heap_locks, d_partial_buffer, d_heap);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpy(h_arr2, d_arr2, n * sizeof(int), cudaMemcpyDeviceToHost));
    for(int i = 0 ;i < n ;i++)
        cout << h_arr2[i] << " " << arr[i]<<"\n";
    cout << "\n";
    bool correct = 1;
    for(int i = 0 ; i < n ; i++)
        if(arr[i] != h_arr2[i])
        correct = 0;
    cout << ((correct)?"Success\n":"Failed!\n");
}

int arr[HEAP_CAPACITY];
int received_arr[HEAP_CAPACITY];
Heap *b = (Heap*)malloc(sizeof(Heap));

void test_insertion()
{
    int n = HEAP_CAPACITY / BATCH_SIZE;
    for(int i = 0 ; i < HEAP_CAPACITY; i++)
        arr[i] = rand() % 90000000 + 10;
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
        td_insertion<<<1, BLOCK_SIZE,0, stream[i%(number_of_streams - 1) + 1]>>>(d_arr + i*BATCH_SIZE, BATCH_SIZE, d_heap_locks, d_partial_buffer, d_heap);
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
    // for(int i = BATCH_SIZE ; i < 2*BATCH_SIZE ; i++)
    //     cout << arr[i] << " " << b ->arr[i] << "\n";
    bool correct = 1;
    for(int i = BATCH_SIZE ; i < 2*BATCH_SIZE ; i++)
        if (arr[i] != b->arr[i] or my_heap.pop() != arr[i])
            correct = 0;
    
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
    sort(b->arr + BATCH_SIZE, b->arr + n*BATCH_SIZE);
    for(int i = BATCH_SIZE ; i < n * BATCH_SIZE ; i++)
        if(b->arr[i] != arr[i])
            correct=0;


    cout << ((correct)?"Success\n":"Failed!\n");
}

void test_deletion() {

    int n = 1e5;
    n = NUMBER_OF_NODES - 1;
    cout<<n<<"\n";
    for(int i = 0 ; i < HEAP_CAPACITY; i++)
        arr[i] = rand() % 9000 + 10;
    // for(int i = 1 ; i < n ; i++)
    // cout << arr[i] << " ";cout << "\n";
    long double total_time = 0;
    int number_of_streams = 5;
    cudaStream_t stream[number_of_streams];
    for(int i = 1 ; i < number_of_streams ; i++)
        cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
    int *d_arr;
    gpuErrchk( cudaMalloc((void**)&d_arr, HEAP_CAPACITY * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr, (arr) , HEAP_CAPACITY * sizeof(int), cudaMemcpyHostToDevice)); 
    cudaDeviceSynchronize();

    std::clock_t c_start = std::clock();
    for(int i = 1; i < n  ; i++)
    {
        td_insertion<<<1, BLOCK_SIZE,0, stream[i%(number_of_streams - 1) + 1]>>>(d_arr + i*BATCH_SIZE, BATCH_SIZE, d_heap_locks, d_partial_buffer, d_heap);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    for(int i = 1 ; i < n; i++) {
        td_delete<<<1, BLOCK_SIZE,0, stream[i%(number_of_streams - 1) + 1]>>>(d_arr, d_heap_locks, d_partial_buffer, d_heap);
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

    gpuErrchk( cudaMemcpy(received_arr, d_arr, HEAP_CAPACITY * sizeof(int), cudaMemcpyDeviceToHost));
    sort(arr + BATCH_SIZE, arr + n * BATCH_SIZE);
    // for(int i = BATCH_SIZE ; i < 2*BATCH_SIZE ; i++)
    //     cout << arr[i] << " " << b ->arr[i] << "\n";
    bool correct = 1;
    for(int i = BATCH_SIZE ; i < n*BATCH_SIZE ; i++) {
        if (arr[i] != received_arr[i]) {
            correct = 0;
            cout << arr[i] << " " << received_arr[i] << " " << i << "\n";
            break;
        }
    }

    cout << ((correct)?"Success\n":"Failed!\n");
}

int main()
{
    heap_init();
    srand(0);
    // srand(time(NULL));
 
    // test_insertion();
    test_deletion();
}