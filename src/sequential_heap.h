#include <iostream>
using namespace std;
// 1-indexed
class CPU_Heap{
    int *arr;
    int capacity;
    int size;
public:
    CPU_Heap(int _capacity);
    bool is_full();
    bool is_empty();
    void push(int key);
    int pop();
};