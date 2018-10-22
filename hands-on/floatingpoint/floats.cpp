//
//  c++ -std=c++14 -Wall -march=native floats.cpp 
//

#include<iostream>
#include<iomanip>
#include<cmath>
#include<limits>
#include<cstdio>
#include<cstring>
#include<bitset>

template<typename T> 
void print(T x) {
 int i; memcpy(&i,&x,sizeof(int)); std::bitset<8*sizeof(T)> bits(i);
 std::cout <<  std::scientific << std::setprecision(8) << x << ' ' <<  std::defaultfloat << x << ' ' 
           << std::hexfloat << x <<' '<< bits << std::endl;
}

int main () {

  std::cout << std::boolalpha;
  std::cout << "Minimum value: " << std::numeric_limits<float>::min() << '\n';
  std::cout << "Maximum value: " << std::numeric_limits<float>::max() << '\n';
  std::cout << "Is signed: " << std::numeric_limits<float>::is_signed << '\n';
  std::cout << "significant bits: " << std::numeric_limits<float>::digits << '\n';
  std::cout << "has infinity: " << std::numeric_limits<float>::has_infinity << '\n';
  std::cout << "base 10 digits: " << std::numeric_limits<float>::digits10 << '\n';
  std::cout << "precision: " << std::numeric_limits<float>::epsilon() << '\n';
  print(std::numeric_limits<float>::epsilon());
  print(0.1f);
  print(std::numeric_limits<float>::min());
  print(std::numeric_limits<float>::max());
  print(std::sqrt(-1.f));
  print(-1.f/0.f);
  print(std::numeric_limits<float>::min()/4.f);
  print(std::pow(2.f,23.f));
  print(std::acos(-1.f));
  print(float(M_PI));
  int i; const float x=0.1f;
  memcpy(&i,&x,sizeof(int));
  std::cout << i << std::endl;
  for (float a=0; a<17; a+=0.5) print(a);
  auto c = std::pow(2.f,23.f);
  print((1.789f+c)-c);
  print((-0.498f+c)-c);

  return 0;
}
