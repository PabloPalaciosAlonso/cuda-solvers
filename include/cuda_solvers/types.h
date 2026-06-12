#pragma once
#include<thrust/complex.h>
#include<thrust/device_vector.h>

namespace cuda_solvers{

#ifndef DOUBLE_PRECISION
  using real = float;
#else
  using real = double;
#endif
  
  using complex = thrust::complex<real>;


  struct OutputInfo{
    bool converged = false;
    int requiredIterations = 0;
    real relativeError = 0.0;
  };

  template<class T>
  struct Result{
    thrust::device_vector<T> x;
    OutputInfo info;
  };
}
