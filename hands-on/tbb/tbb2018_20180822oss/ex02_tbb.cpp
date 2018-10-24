#include <iostream>
#include <vector>
#include <utility>
#include <chrono>
#include <iomanip>
#include <tbb/tbb.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/mutex.h"

int main()
{
  constexpr unsigned int numSteps = 1 << 22;
  unsigned int n = 4;
  tbb::mutex myMutex;
  double totalSum = 0.;
  double pi = 0.;
  constexpr double step = 1.0/(double) numSteps;

  tbb::task_scheduler_init init(n);

  auto start = std::chrono::system_clock::now();
  tbb::parallel_for(tbb::blocked_range<int>(0, numSteps), [&](const tbb::blocked_range<int>& range){
      double partialSum = 0.;
      for(int i = range.begin(); i < range.end(); ++i){
	auto x = (i+0.5)*step;
        partialSum += 4.0/(1.0+x*x);
      }
      tbb::mutex::scoped_lock myLock(myMutex);
      totalSum += partialSum;    
    });

  std::chrono::duration<double> dur= std::chrono::system_clock::now() - start;
  std::cout << "Time spent: " << dur.count() << " seconds" << std::endl;
  pi = totalSum * step;
  std::cout << "Pi: " << pi << std::endl;
}
