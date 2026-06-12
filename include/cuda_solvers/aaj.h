#pragma once
#include "cuda_solvers/types.h"

namespace cuda_solvers::aaj{

  template<class Operator, class T>
  Result<T> solve(const Operator& op,
                  const thrust::device_vector<T>& b,
                  const thrust::device_vector<T>& initialGuess,
                  const Parameters& params,
                  const cudaStream_t st);
}

#include "cuda_solvers/detail/aaj_impl.cuh"
