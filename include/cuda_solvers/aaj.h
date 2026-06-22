#pragma once
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>
#include "cuda_solvers/types.h"

namespace cuda_solvers::aaj{

  struct Parameters {
    int maxIterations          = 5000;
    int memory                 = -1;
    real tolerance             = 1e-4;
    real damping               = 1e-5;
    int notAcceleratedInterval = 2;
    bool verbose               = true;
  };

  template<class Operator,template<class...> class Vec, class T>
  Result<Vec, T> solve(const Operator& op,
                       const Vec<T>& initialGuess,
                       const Parameters& params,
                       cudaStream_t st);
}

#include "cuda_solvers/detail/aaj_impl.cuh"
