
all: test.cu td_ins-td_del.cu sequential_heap.cpp
	nvcc test.cu td_ins-td_del.cu sequential_heap.cpp -Xptxas -O3 -std=c++17 -o test
	./test

clean:
	rm -rf a.out td_ins-td_del test