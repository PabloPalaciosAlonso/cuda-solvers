#pragma once
#include <thrust/execution_policy.h>
#include "cuda_solvers/types.h"


namespace cuda_solvers{
  inline  OutputInfo writeInfo(bool converged, int requiredIterations, real relativeError){
    OutputInfo info;
    info.converged          = converged;
    info.requiredIterations = requiredIterations;
    info.relativeError      = relativeError;
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
