#include <thread>
#include <iostream>
//#include <chrono>

int main()
{
  auto f = [](int i)
    {
      std::cout << "hello world from thread " << i << std::endl;
    };
  //void f(int i);
  std::thread t0(f,0);
  //  std::thread t1(f,1);
  //  std::thread t2(f,2);

  t0.join();
  //  t1.join();
  //  t2.join();
}

// void f(int i)
// {
//   std::cout << "Hello world from thread " << i << std::endl;
// }
