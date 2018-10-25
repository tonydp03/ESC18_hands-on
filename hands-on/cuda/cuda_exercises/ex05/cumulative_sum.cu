#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

float random_float(void)
{

  return static_cast<float>(rand()) / RAND_MAX;
}


// Part 1 of 6: implement the kernel
__global__ void block_sum(const float *input,
                          float *per_block_results,
                          const size_t n)
{
  int sizeVec = blockDim.x;
  __shared__ float sdata[1024];
  int g_index = threadIdx.x + blockDim.x * blockIdx.x;
  int s_index = threadIdx.x;

  sdata[s_index] = input[g_index];
  __syncthreads();

  while (sizeVec!=1){   
    if (s_index < sizeVec/2)
      sdata[s_index] += sdata[sizeVec - 1 - s_index];
    __syncthreads(); 
    sizeVec /=2;   
  }
  if(s_index == 0)
    atomicAdd(per_block_results, sdata[0]);
 
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
  // create array of 256ki elements
  const int num_elements = 1<<18;
  srand(time(NULL));
  // generate random input on the host
  std::vector<float> h_input(num_elements);
  for(int i = 0; i < h_input.size(); ++i)
  {
    h_input[i] = random_float();
  }

  const float host_result = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
  std::cerr << "Host sum: " << host_result << std::endl;
  

  int block_size = 1024;
  int num_blocks = (num_elements + block_size - 1)/block_size;

  //Part 1 of 6: move input to device memory
  float *d_input = 0;
  cudaMalloc(&d_input, num_elements * sizeof(float));
  cudaMemcpy(d_input, &h_input[0], num_elements * sizeof(float), cudaMemcpyHostToDevice);

  // Part 1 of 6: allocate the partial sums: How much space does it need?
  float *d_partial_sums_and_total = 0;
  cudaMalloc(&d_partial_sums_and_total, sizeof(float));

  // Part 1 of 6: launch one kernel to compute, per-block, a partial sum. How much shared memory does it need?
  block_sum<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_input, d_partial_sums_and_total, num_elements);

  // Part 1 of 6: compute the sum of the partial sums
  //block_sum<<<1, num_blocks, num_blocks * sizeof(float)>>>(d_partial_sums_and_total, d_partial_sums_and_total, num_blocks);

  // Part 1 of 6: copy the result back to the host
  float device_result = 0;
  cudaMemcpy(&device_result, d_partial_sums_and_total, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Device sum: " << device_result << std::endl;

  // Part 1 of 6: deallocate device memory
  cudaFree(d_input);
  cudaFree(d_partial_sums_and_total);

  return 0;
}
