#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c);
std::vector<int> make_vector(int N);

int main()
{
  // create a vector of N elements, generated randomly
  int const N = 10;
  std::vector<int> v = make_vector(N);
  std::cout << v << '\n';

  // multiply all the elements of the vector
  // use std::accumulate
  auto product = std::accumulate(v.begin(), v.end(), 1LL, std::multiplies<>());
  std::cout << "Product: " << product << '\n';

  // sort the vector in descending order
  // use std::sort
  std::sort(v.begin(), v.end(), [](int a, int b) {return a > b;});
  std::cout << "Sorted vector: " << v << '\n';

  // move the even numbers at the beginning of the vector
  // use std::partition
  std::partition(v.begin(), v.end(), [](int a){return !(a % 2);});
  std::cout << "Partitioned vector: " << v << '\n';

  // create another vector with the squares of the numbers in the first vector
  // use std::transform
  std::vector<int> v2;
  std::transform(v.begin(), v.end(), std::back_inserter(v2), [](int n) {return n * n;});
  std::cout << "Transformed vector: " << v2 << '\n';

  // find the first multiple of 3 or 7
  // use std::find_if
  auto multiple = std::find_if(v.begin(), v.end(), [](int n) { return n % 3 == 0 || n % 7 == 0; });
  std::cout << "Primo multiplo di 3 o 7: " << *multiple << " at position: " << std::distance(v.begin(), multiple) << '\n';

  // erase from the vector all the multiples of 3 or 7
  // use std::remove_if followed by vector::erase

  v.erase(std::remove_if(v.begin(), v.end(), [](int n) { return (n % 3) == 0 || (n % 7) == 0; }), v.end());
  std::cout << "New vector: " << v << std::endl;

};

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c)
{
  os << "{ ";
  std::copy(
            std::begin(c),
            std::end(c),
            std::ostream_iterator<int>{os, " "}
            );
  os << '}';

  return os;
}

std::vector<int> make_vector(int N)
{
  // define a pseudo-random number generator engine and seed it using an actual
  // random device
  std::random_device rd;
  std::mt19937 eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> dist{1, MAX_N};

  std::vector<int> result;
  result.reserve(N);
  std::generate_n(std::back_inserter(result), N, [&]() { return dist(eng); });

  return result;
}
