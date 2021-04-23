# Accelerating Concurrent Heap on GPUs

This code is implementation of algortihm described in the following paper:

Yan-Hao Chen, Fei Hua, C. Huang, Jeremy Bierema, Chi Zhang, and E. Z. Zhang. 2019. Accelerating Concurrent Heap on GPUs.ArXivabs/1906.06504 (2019).  https://arxiv.org/abs/1906.06504

This was my project for the CSE-560 course on GPU computing offered by IIIT-D under the guidance of **Dr. Ojaswa Sharma**.

How to run:

``` make``` in src/ directory and it will compile, link and run the test file automatically.

# Results:

Excellent speedups were obtained in comparison to CPU heap implementation and speedups are as high as 130x when compared STL-heap on inserted approximately 130 million keys in the heap.

Time graph on varying number of keys inserted:
![](https://github.com/CSE-560-GPU-Computing-2021/project_-team_15/blob/master/results/graph1.png)

<br>

Speedup graph on varying number of keys inserted. Speedup is with respect to my naive CPU implementation of heap:
![](https://github.com/CSE-560-GPU-Computing-2021/project_-team_15/blob/master/results/graph2.png)
<br>

Speedup graph on varying number of keys inserted. Speedup is with respect to **STL-HEAP**:
![](https://github.com/CSE-560-GPU-Computing-2021/project_-team_15/blob/master/results/graph2.2.png)
