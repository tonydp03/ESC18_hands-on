#ifndef Data_h
#define Data_h

#include<cstdint>
#include<vector>
#include<memory>



using Float=float;

enum Quality: uint8_t { bad, loose, strict, tight };

struct Data {

  Data(){}
  // virtual ~Data(){}
  
  Float x,y,z;
  Float vx,vy,vz;
  bool isValid;
  Quality quality;

};

using AOS = std::vector<Data>;
using AOP = std::vector<std::unique_ptr<Data>>;


struct SOA {
  Float *x, *y, *z;
  Float *vx, *vy, *vz;
  bool *isValid;
  Quality *quality;
  uint32_t size;
};



struct VSOA {
  explicit VSOA(uint32_t s) :
    x(s),y(s),z(s),
    vx(s),vy(s),vz(s),
    isValid(s),quality(s){}
  
  std::vector<Float> x, y, z;
  std::vector<Float> vx, vy, vz;
  std::vector<bool> isValid;
  std::vector<Quality> quality;  
};






#endif 
