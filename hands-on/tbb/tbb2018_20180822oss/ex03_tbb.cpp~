#include <thread>
#include <iostream>
#include <vector>
#include <utility>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>

int main()
{
  constexpr unsigned int N = 1 << 23;
  unsigned int n = 4;
  std::mutex myMutex;
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

  auto f = [&](int threadNumber)
    {
      int partialNi = 0;
      auto start =  threadNumber * (N/n);
      auto end = start + (N/n);
      if (threadNumber == n - 1)
  	end = N;
      for (int j = start; j < end; j++){ 
	auto r = sqrt(pow(inputX[j], 2) + pow(inputY[j], 2)); 
	if(r < 1)
	  partialNi++;
      }
      std::lock_guard<std::mutex> myLock(myMutex);
      totalNi += partialNi;
    };

  std::vector<std::thread> v;
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < n; ++i) {
    v.emplace_back(f, i);
  };  

  for (auto& t : v) {
    t.join();
  };

  std::chrono::duration<double> dur= std::chrono::system_clock::now() - start;
  std::cout << "Time spent: " << dur.count() << " seconds" << std::endl;
  pi = ((float) totalNi/N) * 4;
  std::cout << "Pi: " << pi << std::endl;
}
