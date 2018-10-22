#include<random>
#include<iostream>
#include<fstream>
#include<iomanip>
#include<array>
#include<cmath>
#include<cassert>
#include <chrono>
#include "../architecture/benchmark.h"
#include "simpleSinCos.h"

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



  struct Point {
    float phi,r;
#ifdef THETA
    float t;
#endif
  };
  struct Points {
    std::array<Point,N> p;
  };


  Points points;



  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;


  //outer loop just to make timing "macroscopical"
  for (int j=0; j<1000; ++j) {

    for(auto & p : points.p) {
      p.phi = -M_PI +2.*M_PI*rgen(eng);
      p.r =  rgen(eng);
    }
    constexpr int NBin = 100;
    struct Hist {
      int bin[NBin+1][NBin+1] = {0};
    };
    float binWidthI = NBin/2.;

    Hist h;

    delta -= (std::chrono::high_resolution_clock::now()-start);
    benchmark::touch(points);
    // the real loop
    int stride=16;
    for(int i=0; i<N; i+=stride) {  // eventually account for N not being a multiple of stride....
     int xbin[stride], ybin[stride];
     for (int k=0; k<stride;++k) {
       auto const & p = points.p[i+k];
       float s,c; simpleSinCos(p.phi,s,c);
       auto x = p.r*c;
       auto y = p.r*s;
       xbin[k] = (x+1.f)*binWidthI;
       ybin[k] = (y+1.f)*binWidthI;
     }
    // assert(xbin>=0 && ybin>=0);
     // assert(xbin<101 && ybin<101);
     for (int k=0; k<stride;++k) ++h.bin[xbin[k]][ybin[k]];
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
