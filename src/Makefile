
test: test.cu td_ins-td_del.cu
	nvcc test.cu -Xptxas -O2 td_ins-td_del.cu sequential_heap.cpp -std=c++17 -o test
	./test
all:
	nvcc td_ins-td_del.cu -o td_ins-td_del

library: 
	nvcc -shared -std=c++17 -Xcompiler -fPIC td_ins-td_del.cu -o libtd_ins-td_del.so


clean:
	rm -rf a.out td_ins-td_del test libtd_ins-td_del.so