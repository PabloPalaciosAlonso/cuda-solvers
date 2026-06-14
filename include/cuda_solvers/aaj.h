#pragma once
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>
#include "cuda_solvers/types.h"

namespace cuda_solvers::aaj{

  struct Parameters {
    int maxIterations;
    int memory;
    real tolerance;
    real damping;
    int notAcceleratedInterval;
  };

  template<class Operator, class T>
  Result<T> solve(const Operator& op,
                  const thrust::device_vector<T>& initialGuess,
                  const Parameters& params,
                  cudaStream_t st);
}

#include "cuda_solvers/detail/aaj_impl.cuh"
