#include <stdio.h>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0
__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  cudaSetDevice(MYDEVICE);
  int N = 20 * (1 << 25);
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start);

  saxpy<<<(N+511)/512, 512>>>(N, 2.0f, d_x, d_y);

  cudaEventRecord(stop);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);


  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0f));
  }
  
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  float memRate = 3004 * 1.e6; //Hz
  int memBusWidth = 48; //bytes
  float thThroughput = (memRate * memBusWidth * 2.)/1.e9;
  printf("Theoretical Throughput (GB/s): %f\n", thThroughput);

  float totalData = 3 * N * sizeof(float);
  float acThroughput = totalData/(milliseconds *1.e6);
  printf("Actual Throughput (GB/s): %f\n", acThroughput);

}



