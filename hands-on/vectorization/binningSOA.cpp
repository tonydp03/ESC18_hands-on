#include<random>
#include<iostream>
#include<fstream>
#include<iomanip>
#include<array>
#include<cmath>
#include<cassert>
#include <chrono>
#include "../architecture/benchmark.h"


std::mt19937 eng;
std::mt19937 eng2;
std::uniform_real_distribution<double> rgen(0.,1.);

template<typename T>
void put(std::ostream& co, T x) {
  unsigned const char * out = (unsigned const char *)(&x);
  for(int i=0; i<sizeof(T); ++i) co<< out[i];
}


int main() {

  constexpr int N=1<<14;
  std::cout << "working with batch of " << N << " particles" << std::endl;


  struct P {
    std::array<float,N> phi,r;
  };


  P p;
  
  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;


  //outer loop just to make timing "macroscopical"
  for (int j=0; j<1000; ++j) {

    for(int i=0; i<N; ++i) {
      p.phi[i] = -M_PI +2.*M_PI*rgen(eng);
      p.r[i] =  rgen(eng);
    }
    constexpr int NBin = 100;
    struct Hist {
      int bin[NBin+1][NBin+1] = {0};
    };
    float binWidth = 2./NBin;

    Hist h;

    delta -= (std::chrono::high_resolution_clock::now()-start);
    benchmark::touch(p);
    // the real loop
    for(int i=0; i<N; ++i) {
     auto x = p.r[i]*std::cos(p.phi[i]);
     auto y = p.r[i]*std::sin(p.phi[i]);
     int xbin = (x+1.f)/binWidth;
     int ybin = (y+1.f)/binWidth;
    // assert(xbin>=0 && ybin>=0);
     // assert(xbin<101 && ybin<101);
     ++h.bin[xbin][ybin];
    }  
    benchmark::keep(h);
    delta += (std::chrono::high_resolution_clock::now()-start);
//    std::cout << '.';
   }
   std::cout <<" Computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
              << " ms" << std::endl;

   return 0;
}
