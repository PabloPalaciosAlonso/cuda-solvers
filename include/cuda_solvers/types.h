#pragma once
#include<thrust/complex.h>

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

  template<template<class...> class Vec, class T>
  struct Result{
    Vec<T> x;
    OutputInfo info;
  };
}
