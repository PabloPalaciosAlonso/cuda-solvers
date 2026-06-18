#pragma once
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include"cuda_solvers/types.h"

namespace cuda_solvers{

  #define CUSOLVER_CALL(x)                                      \
  do {                                                        \
    cusolverStatus_t s = (x);                                 \
    if (s != CUSOLVER_STATUS_SUCCESS)                         \
      throw std::runtime_error("cuSOLVER error");             \
  } while (0)

#define CUDA_CALL(x)                                          \
  do {                                                        \
    cudaError_t e = (x);                                      \
    if (e != cudaSuccess)                                     \
      throw std::runtime_error(cudaGetErrorString(e));        \
  } while (0)

#ifdef DOUBLE_PRECISION
  using cucomplex = cuDoubleComplex;
#define cusolverDnComplexgels cusolverDnZZgels
#define cusolverDnComplexgels_bufferSize cusolverDnZZgels_bufferSize
#else
  using cucomplex = cuComplex;
#define cusolverDnComplexgels cusolverDnCCgels
#define cusolverDnComplexgels_bufferSize cusolverDnCCgels_bufferSize
#endif

  template<class...> class Vec>
  struct LSWorkspace {
    cusolverDnHandle_t solver = nullptr;

    Vec<complex> A_tmp;
    Vec<complex> b_tmp;
    Vec<unsigned char> work;
    Vec<int> info;

    int rows = 0;
    int cols = 0;
    size_t work_bytes = 0;

    LSWorkspace(int rows_, int cols_)
      : rows(rows_), cols(cols_)
    {
      CUSOLVER_CALL(cusolverDnCreate(&solver));

      A_tmp.resize(size_t(rows) * size_t(cols));
      b_tmp.resize(rows);
      info.resize(1);

      size_t needed = 0;

      CUSOLVER_CALL(cusolverDnComplexgels_bufferSize(solver,
                                                     rows,
                                                     cols,
                                                     1,
                                                     nullptr,
                                                     rows,
                                                     nullptr,
                                                     rows,
                                                     nullptr,
                                                     cols,
                                                     nullptr,
                                                     &needed));
      
      work.resize(needed);
      work_bytes = needed;
    }
    
    ~LSWorkspace() {
      if (solver)
        cusolverDnDestroy(solver);
    }

    LSWorkspace(const LSWorkspace&) = delete;
    LSWorkspace& operator=(const LSWorkspace&) = delete;
  };


  template<class...> class Vec>
  inline void solve_least_squares(Vec<complex>& A,
                                  Vec<complex>& b,
                                  Vec<complex>& out,
                                  LSWorkspace& ws,
                                  const cudaStream_t& st){
    const int rows = ws.rows;
    const int cols = ws.cols;
    
    assert(rows >= cols);
    assert(A.size() >= rows * cols);
    assert(b.size() >= rows);
    assert(out.size() >= cols);
    
    CUSOLVER_CALL(cusolverDnSetStream(ws.solver, st));
    
    auto A_src = thrust::raw_pointer_cast(A.data());
    auto b_src = thrust::raw_pointer_cast(b.data());
    auto A_dst = thrust::raw_pointer_cast(ws.A_tmp.data());
    auto b_dst = thrust::raw_pointer_cast(ws.b_tmp.data());
    auto x_dst = thrust::raw_pointer_cast(out.data());
    
    CUDA_CALL(cudaMemcpyAsync(A_dst,
                              A_src,
                              size_t(rows) * size_t(cols) * sizeof(complex),
                              cudaMemcpyDeviceToDevice,
                              st));
    
    CUDA_CALL(cudaMemcpyAsync(b_dst,
                              b_src,
                              size_t(rows) * sizeof(complex),
                              cudaMemcpyDeviceToDevice,
                              st));
    
    auto dA = reinterpret_cast<cucomplex*>(A_dst);
    auto dB = reinterpret_cast<cucomplex*>(b_dst);
    auto dX = reinterpret_cast<cucomplex*>(x_dst);

    auto dWork = static_cast<void*>(thrust::raw_pointer_cast(ws.work.data()));
    
    auto dInfo = thrust::raw_pointer_cast(ws.info.data());
    
    int niter = 0;
    
    CUSOLVER_CALL(cusolverDnComplexgels(ws.solver,
                                        rows,
                                        cols,
                                        1,
                                        dA,
                                        rows,
                                        dB,
                                        rows,
                                        dX,
                                        cols,
                                        dWork,
                                        ws.work_bytes,
                                        &niter,
                                        dInfo));

    int info_h = 0;
    CUDA_CALL(cudaMemcpyAsync(&info_h,
                              dInfo,
                              sizeof(int),
                              cudaMemcpyDeviceToHost,
                              st));
    
    CUDA_CALL(cudaStreamSynchronize(st));
    
    if (info_h != 0) {
      throw std::runtime_error("cuSOLVER gels failed: devInfo != 0");
    }
  }
}
