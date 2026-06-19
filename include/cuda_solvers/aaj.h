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
    bool verbose = true;
  };

  template<class Operator,template<class...> class Vec, class T>
  Result<Vec, T> solve(const Operator& op,
                       const Vec<T>& initialGuess,
                       const Parameters& params,
                       cudaStream_t st);
}

#include "cuda_solvers/detail/aaj_impl.cuh"
