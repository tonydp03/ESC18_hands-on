//
//  c++ -O2 grouping.cpp memory_usage.cc /usr/local/Cellar/jemalloc/5.1.0/lib/libjemalloc.dylib
//
#include<random>
#include<vector>
#include<cstdint>
#include<algorithm>
#include<iostream>
#include<cassert>

#include<chrono>

#include "memory_usage.h"

auto start = std::chrono::high_resolution_clock::now();

uint64_t maxLive=0;

void stop(const char * m) {
  auto delta = std::chrono::high_resolution_clock::now()-start;
  maxLive= std::max( maxLive, memory_usage::totlive() );
  std::cout << m;
  std::cout << " elapsted time (ms) " << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << std::endl;
  std::cout << "allocated so far " << memory_usage::allocated();
  std::cout << " deallocated so far " << memory_usage::deallocated() << std::endl;
  std::cout << "total / max live " << memory_usage::totlive() << ' ' << maxLive << std::endl;

  start = std::chrono::high_resolution_clock::now();
}

constexpr int NG=100000;
constexpr int ME=80;

std::mt19937 reng;
std::poisson_distribution<int> aGen(NG);
std::poisson_distribution<int> bGen(ME);


class Generator {
public:

  void generate(bool doprint) {
    // generate ng groups
    ng = aGen(reng);

    // for each group generate elements
    int ne[ng];
    totElements=0;
    for(auto i=0U;i<ng;++i) totElements += (ne[i]=bGen(reng));
    if (doprint) std::cout << "--- Generated " << totElements << " in " << ng << " groups" << std::endl;
  }


  // return to which "initial group" element i has been found...
  uint32_t protoGroup(uint32_t i) const {
    i = i%totElements;
    i = i%ng + (i/7)%ng; // add a "beat"  (some groups may be empty...)
    return i%ng;
    
  }

  // return in which subgroup of "g" "i" belongs  ( 0 or 1)
  uint32_t split(uint32_t g, uint32_t i) const {
    // assert(protoGroup(i)==g);
    return  (g%1014) ? (i%3)/2 : 0;      
  }

  auto nElements() const {
    return totElements;
  }

  // fake weight
  float weight(uint32_t el) const {
    // assert(el< totElements);
    return 0.01f*float(el);
  }
  
 private:
  uint32_t ng;
  uint32_t totElements;
  
  
};



Generator generator;

#include<map>  // not necessarely the best choice!
#include<unordered_map>  // not necessarely the best choice!
void one(bool doprint) {
 if (doprint) stop("before generation");

 generator.generate(doprint);

 auto ntot = generator.nElements();

  if (doprint) stop("aftert generation");

  // here you have to find the first set of groups
  // in principle you are allowed to call generator.protoGroup only ONCE per element as it is faking an clustering algo

  // this is just a test to verify the generator
  //  std::unordered_map<int,int> count;  // in std the default constructor of int IS int(0)
  std::map<int,int> count;  // in std the default constructor of int IS int(0)
  for (auto i=0U;i<ntot;++i) ++count[generator.protoGroup(i)];
  if (doprint) std::cout << "--- Found " << count.size() << " proto-groups" << std::endl;

  if (doprint) stop("aftert protoGroups");

  
  // and then split them in the final set

  // again a test of consistency....
  for (auto i=0U;i<ntot;++i) {
    auto sub = generator.split(generator.protoGroup(i),i); // no, you are not allowed to use generator.protoGroup here!!
    assert(sub<2);
  }
  if (doprint) stop("after splitting");


  // at the end you should be able to tell with element is in which final group...
  // for instance demonstrate you can compute the total "weight" of a group calling generator.weight(j);

  
 if (doprint) stop("end of algo");
 
}


int main() {

  one(true);
  stop("\nafter call: ");

  for (int i=0; i<20; ++i) {
    one(false);
    // stop("after call");
  }
  stop("\nafter loop in main: ");

  one(true);

  stop("\nat the end: ");

  
  return 0;

}





