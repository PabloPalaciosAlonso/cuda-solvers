#pragma once

#include "cuda_solvers/types.h"


namespace cuda_solvers::gmres{
  
  struct Parameters {
    int  maxIterations = 300;
    int  memory        = 30;
    real tolerance     = real(1e-6);
    bool verbose       = true;
  };
  
  template<class Operator,template<class...> class Vec, class T>
  Result<Vec, T> solve(const Operator& op,
                       const Vec<T>& b,
                       const Vec<T>& initialGuess,
                       const Parameters& params,
                       cudaStream_t st);
}
#include "cuda_solvers/detail/gmres_impl.cuh"
