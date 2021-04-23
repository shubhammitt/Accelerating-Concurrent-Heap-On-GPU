/*
 * Author: Shubham Mittal
 * IIITD, 2018101
 * GPU Project: Accelerating Concurrent Heap on GPUs
*/
#include "sequential_heap.h"

CPU_Heap::CPU_Heap(int _capacity) {
    capacity = _capacity + 1;
    arr = new int[capacity];
    size = 0;
}

bool CPU_Heap::is_full() {
    return size + 1 == capacity;
}

bool CPU_Heap::is_empty() {
    return size == 0;
}

void CPU_Heap::push(int key) {
    if (is_full()) {
        cout << "Heap Full!\n";
        return;
    }
    ++size;
    int par = size >> 1;
    int child = size;
    arr[child] = key;
    while(child != 1) {
        if(arr[par] <= arr[child])
            return;
        swap(arr[par], arr[child]);
        child = par;
        par >>= 1;
    }
}

int CPU_Heap::pop() {
    if (is_empty()) {
        cout << "Heap empty!";
        return -1;
    }
    int key = arr[1];
    arr[1] = arr[size--];
    int par = 1;
    int left_child = par << 1;
    int right_child = left_child + 1;
    int idx = 0;
    while(left_child <= size) {
        idx = left_child;
        if(right_child <= size and arr[left_child] > arr[right_child])
            idx = right_child;
        if(arr[par] > arr[idx]) {
            swap(arr[par], arr[idx]);
            par = idx;
            left_child = par << 1;
            right_child = left_child + 1;
        }
        else
            break;
    }
    return key;
}

// int main(int argc, char *argv[]) {
//     int n = 5;
//     if(argc > 1) n = atoi(argv[1]);
//     CPU_Heap obj(n);
//     for(int i = 0 ; i < n; i++) {
//         int x = rand() % 10;
//         obj.push(x);
//         cout << x << " ";
//     }
//     cout << "\n";
//     for(int i = 0; i < n ; i++) {
//         cout << obj.pop() << " ";
//     }
//     cout << "\n";
// }