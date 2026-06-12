#include "cuda_solvers/types.h"


namespace cuda_solvers{
  inline  OutputInfo writeInfo(bool converged, int requiredIterations, real relativeError){
    OutputInfo info;
    info.converged          = converged;
    info.requiredIterations = requiredIterations;
    info.relativeError      = relativeError;
    return info;
  }

  
  inline int index2D(int i, int j, int ncols) {
    return i * ncols + j; // row-major
  }
  
  template <class T>
  inline thrust::device_vector<T> getColumn(const thrust::device_vector<T>& matrix,
                                            thrust::device_vector<T>& out,
                                            const int col,
                                            const cudaStream_t st) {
    
    int N = out.size();
    thrust::copy(thrust::device,
                 matrix.begin() + col * N,
                 matrix.begin() + (col + 1) * N,
                 out.begin());
    return out;
  }
  
  template <class T>
  inline void writeColumn(const thrust::device_vector<T>& v,
                          thrust::device_vector<T>& V,
                          const int N,
                          const int col,
                          const cudaStream_t st) {
    
    assert((int)v.size() == N);
    
    thrust::copy(thrust::device,
                 v.begin(), v.end(),
                 V.begin() + col * N);
  }
}
