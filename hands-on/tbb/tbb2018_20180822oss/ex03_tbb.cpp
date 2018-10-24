#include <iostream>
#include <vector>
#include <utility>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include <tbb/tbb.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/mutex.h"

int main()
{
  constexpr unsigned int N = 1 << 23;
  unsigned int n = 4;
  tbb::mutex myMutex;
  int totalNi = 0;
  double pi = 0.;

  std::vector<float> inputX;
  inputX.reserve(N);
  std::vector<float> inputY;
  inputY.reserve(N);

  std::mt19937 engine(time(0));
  std::uniform_real_distribution<> uniformDist(-1, 1);

  for (unsigned int k = 0; k < 2 * N; ++k)
    k >= N ? inputX.emplace_back(uniformDist(engine)) : inputY.emplace_back(uniformDist(engine));

  auto start = std::chrono::system_clock::now();
  tbb::parallel_for(tbb::blocked_range<int>(0, N), [&](const tbb::blocked_range<int>& range){
      int partialNi = 0;
      for(int i = range.begin(); i < range.end(); ++i){
	auto r = sqrt(pow(inputX[i], 2) + pow(inputY[i], 2)); 
	if(r < 1)
	  partialNi++;
      }
      tbb::mutex::scoped_lock myLock(myMutex);
      totalNi += partialNi;
    });

  std::chrono::duration<double> dur= std::chrono::system_clock::now() - start;
  std::cout << "Time spent: " << dur.count() << " seconds" << std::endl;
  pi = ((float) totalNi/N) * 4;
  std::cout << "Pi: " << pi << std::endl;
}
