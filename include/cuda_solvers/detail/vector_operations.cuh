#pragma once
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include"cuda_solvers/types.h"

namespace cuda_solvers{

  template<template<class...> class Vec, class T>
  inline void substract(const Vec<T>& v1,
                        const Vec<T>& v2,
                        Vec<T>& out,
                        cudaStream_t st = 0) {
    
    assert(v1.size() == v2.size());
    assert(out.size() == v1.size());

    auto policy = thrust::cuda::par.on(st);
    thrust::transform(policy,
                      v1.begin(), v1.end(),
                      v2.begin(),
                      out.begin(),
                      thrust::minus<T>());
  }

  template<template<class...> class Vec, class T>
  inline void add(const Vec<T>& v1,
                  const Vec<T>& v2,
                  Vec<T>& out,
                  cudaStream_t st = 0) {
    
    assert(v1.size() == v2.size());
    assert(out.size() == v1.size());
    auto policy = thrust::cuda::par.on(st);
    
    thrust::transform(policy,
                      v1.begin(), v1.end(),
                      v2.begin(),
                      out.begin(),
                      thrust::plus<T>());
  }

  template<template<class...> class Vec, class T>
  inline void multiply(const Vec<T>& v1,
                       const Vec<T>& v2,
                       Vec<T>& out,
                       cudaStream_t st = 0) {

    assert(v1.size() == v2.size());
    assert(out.size() == v1.size());

    auto policy = thrust::cuda::par.on(st);
    thrust::transform(policy,
                      v1.begin(), v1.end(),
                      v2.begin(),
                      out.begin(),
                      thrust::multiplies<T>());
  }

  template<template<class...> class Vec, class T>
  inline void multiply(const Vec<T>& v,
                       const T scalar,
                       Vec<T>& out,
                       const cudaStream_t st = 0) {

    auto policy = thrust::cuda::par.on(st);

    assert(out.size() == v.size());
    thrust::transform(policy,
                      v.begin(), v.end(),
                      thrust::make_constant_iterator(scalar),
                      out.begin(),
                      thrust::multiplies<T>());
  }

  template<template<class...> class Vec, class T>
  inline void divide(const Vec<T>& v1,
                     const Vec<T>& v2,
                     Vec<T>& out,
                     const cudaStream_t st = 0) {
  
    assert(v1.size() == v2.size());
    assert(out.size() == v1.size());
  
    auto policy = thrust::cuda::par.on(st);
    thrust::transform(policy,
                      v1.begin(), v1.end(),
                      v2.begin(),
                      out.begin(),
                      thrust::divides<T>());
  }

  template<template<class...> class Vec, class T>
  inline void divide(const Vec<T>& v,
                     const T scalar,
                     Vec<T>& out,
                     const cudaStream_t st = 0) {
    multiply(v, T(1.0)/scalar, out, st);
  }
     
  template<class T>
  struct dotProductFunctor;
  
  template<>
  struct dotProductFunctor<real> {
    __host__ __device__ real operator()(const real& x, const real& y) const {
      return x * y;
    }
  };
  
  template<>
  struct dotProductFunctor<complex> {
    __host__ __device__ complex operator()(const complex& x, const complex& y) const {
      return conj(x) * y;
    }
  };

  inline __host__ __device__ real make_real(const real& x) {
    return x;
  }
  
  inline __host__ __device__ real make_real(const complex& x) {
    return x.real();
  }
  
  template<template<class...> class Vec, class T>
  inline T dotc(const Vec<T>& v1,
                const Vec<T>& v2,
                const cudaStream_t st = 0) {
    
    assert(v1.size() == v2.size());
    auto policy = thrust::cuda::par.on(st);
    return thrust::inner_product(policy,
                                 v1.begin(), v1.end(),
                                 v2.begin(),
                                 T(),
                                 thrust::plus<T>(),
                                 dotProductFunctor<T>());
  }

  template <template<class...> class Vec, class T>
  inline real norm(const Vec<T> &vec, const cudaStream_t st = 0)
  {
    T norm2 = dotc(vec, vec, st);
    return std::sqrt(make_real(norm2));
  }

  template<class T>
  struct AxpyFunctor {
    T alpha;
    
    __host__ __device__
    T operator()(const T& yv, const T& xv) const {
      return yv + alpha * xv;
    }
  };
  
  template<template<class...> class Vec, class T>
  void axpy(const Vec<T>& x,
            const Vec<T>& y,
            Vec<T>& out,
            const T alpha,
            const cudaStream_t st = 0) {
    assert(x.size() == y.size());
    assert(out.size() == y.size());
    auto policy = thrust::cuda::par.on(st);
    thrust::transform(policy,
                      y.begin(), y.end(),
                      x.begin(),
                      out.begin(),
                      AxpyFunctor<T> {alpha});   
  }
}
