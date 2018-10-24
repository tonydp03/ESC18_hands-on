#include <tbb/tbb.h>
#include "tbb/task_scheduler_init.h"
#include <iostream>

int main()
{
  tbb::task_scheduler_init init;
  std::cout << "Hello World!" << std::endl;
}
