#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cassert>

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c);
std::vector<int> make_vector(int N);

int main()
{
  // create a vector of N elements, generated randomly
  int const N = 10;
  std::vector<int> v = make_vector(N);
  std::cout << v << '\n';

  // sum all the elements of the vector
  // use std::accumulate
  auto sum = std::accumulate(v.begin(), v.end(), 0);
  std::cout << "Sum is " << sum << '\n';

  // compute the average of the first half and of the second half of the vector
  auto midIt = v.begin() + v.size()/2;
  auto avg1 = 1. * std::accumulate(v.begin(), midIt, 0)/std::distance(v.begin(), midIt);  
  auto avg2 = 1. * std::accumulate(midIt, v.end(), 0)/std::distance(midIt, v.end());  
  std::cout << "Average 1: " << avg1 << '\n';
  std::cout << "Average 2: " << avg2 << '\n';

  // move the three central elements to the beginning of the vector
  // use std::rotate
  assert(v.size() >= 3);
  auto elFirst = std::prev(midIt);
  auto elLast = std::next(midIt,2);
  std::rotate(v.begin(), elFirst, elLast);

  // remove duplicate elements
  // use std::sort followed by std::unique/unique_copy
  // then remove consecutive duplicate elements                                                                                                                                  
  std::sort(v.begin(), v.end());
  std::vector<int> unique_v;
  std::unique_copy(v.begin(), v.end(), std::back_inserter(unique_v));
  std::cout << "unique: " << unique_v << '\n';

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
  std::random_device rd;
  std::mt19937 eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> dist{1, MAX_N};

  std::vector<int> result;
  result.reserve(N);
  std::generate_n(std::back_inserter(result), N, [&]() { return dist(eng); });

  return result;
}
