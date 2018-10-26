#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>

#define MYDEVICE 0
#define NUM_BINS 4096


__global__ void computeHistogram(int *input, unsigned int *histogram, int dim)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index < dim){
    int number = input[index];
    atomicAdd(&histogram[number], 1);
    if(histogram[number] > 127)
      histogram[number] = 127;
  }
}

int main(void){

  // create array of input elements
  const int num_elements = 1 << 19;
  srand(time(NULL));
  // generate random input on the host
  std::vector<int> h_input;
  for(int i = 0; i < num_elements; i++)
    h_input.push_back(rand() % NUM_BINS);

  int inputSize = num_elements * sizeof(int);
  int *d_input = 0;
  cudaMalloc(&d_input, inputSize);
  cudaMemcpy(d_input, &h_input[0], inputSize, cudaMemcpyHostToDevice);

  int histSize = NUM_BINS * sizeof(unsigned int);
  unsigned int *h_hist = 0;
  unsigned int *d_hist = 0;
  cudaMallocHost(&h_hist, histSize);
  cudaMalloc(&d_hist, histSize);

  int block_size = 1024;
  int num_blocks = (num_elements + block_size - 1)/block_size;

  computeHistogram<<<num_blocks, block_size>>>(d_input, d_hist, num_elements);

  cudaMemcpy(h_hist, d_hist, histSize, cudaMemcpyHostToDevice);

  for (int i = 0; i < NUM_BINS; i++)
    std::cout << "Bin: " << i << " Elements: " << h_hist[i] << '\n';
 
  cudaFree(d_input);
  cudaFree(h_hist);
  cudaFree(d_hist);

  return 0;
}
