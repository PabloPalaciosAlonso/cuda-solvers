#pragma once
#include "cuda_solvers/types.h"

namespace cuda_solvers::gmres{

  struct Parameters {
    int  maxIterations = 300;
    int  memory        = 30;
    real tolerance     = real(1e-6);
  };
  
  template<class Operator, class T>
  Result<T> solve(const Operator& op,
                  const thrust::device_vector<T>& b,
                  const thrust::device_vector<T>& initialGuess,
                  const Parameters& params,
                  cudaStream_t st);
}
#include "cuda_solvers/detail/gmres_impl.cuh"
