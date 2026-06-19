#pragma once
#include <thrust/execution_policy.h>
#include "cuda_solvers/types.h"
#include "cuda_solvers/detail/logger.h"

namespace cuda_solvers{
  inline  OutputInfo writeInfo(bool converged, int requiredIterations,
                               real relativeError, bool verbose,
                               std::string method){
    OutputInfo info;
    info.converged          = converged;
    info.requiredIterations = requiredIterations;
    info.relativeError      = relativeError;

    if (verbose) {
      if (converged) {
        LOG_INFO("[" << method << "] Successfully converged");
        LOG_INFO("[" << method << "] Relative error: " << relativeError);
        LOG_INFO("[" << method << "] Required iterations: " << requiredIterations);
      } else {
        LOG_WARN("[" << method << "] Did not converge");
        LOG_WARN("[" << method << "] Relative error: " << relativeError);
        LOG_WARN("[" << method << "] Number of iterations: " << requiredIterations);
      }
    }
    return info;
  }
  
  
  inline int index2D(int i, int j, int nrows) {
    return i + nrows * j;
  }
  
  template <template<class...> class Vec, class T>
  inline void getColumn(const Vec<T>& matrix,
                        Vec<T>& out,
                        const int col,
                        const cudaStream_t st) {
    
    int N = out.size();
    thrust::copy(thrust::cuda::par.on(st),
                 matrix.begin() + col * N,
                 matrix.begin() + (col + 1) * N,
                 out.begin());
  }
  
  template <template<class...> class Vec, class T>
  inline void writeColumn(const Vec<T>& v,
                          Vec<T>& V,
                          const int N,
                          const int col,
                          const cudaStream_t st) {
    
    assert((int)v.size() == N);
    
    thrust::copy(thrust::cuda::par.on(st),
                 v.begin(), v.end(),
                 V.begin() + col * N);
  }
}
