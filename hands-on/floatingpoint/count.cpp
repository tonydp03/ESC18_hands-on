#include<iostream>
#include<iomanip>
#include<cmath>
#include<limits>
#include<cstdio>
#include<cstring>

int count(float x, float e) {
  int c=0;
  while(x<e) {x=std::nextafter(x,2*e); ++c;}
  return c;
}

int diff(float x, float e) {
  int ix=0; memcpy(&ix,&x,sizeof(int));
  int ie=0; memcpy(&ie,&e,sizeof(int));
  return ie-ix;
}
int main() {


  std::cout << count(.1f,1.f) << std::endl;
  std::cout << count(2.f,3.f) << std::endl;
  std::cout << diff(.1f,1.f) << std::endl;
  std::cout << diff(2.f,3.f) << std::endl;
		     
  return 0;

};
