#pragma once
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include "cuda_solvers/types.h"
#include "cuda_solvers/gmres.h"
#include "cuda_solvers/detail/vector_operations.cuh"
#include "cuda_solvers/detail/least_squares.cuh"
#include "cuda_solvers/detail/utils.h"
#include "cuda_solvers/detail/logger.h"


namespace cuda_solvers::aaj{

  namespace detail{

    inline Parameters setDefaultMemory(const Parameters& p_in, int N){
      Parameters p = p_in;
      if (p.memory <= 0) {
        p.memory = std::min(N / 2 + 1, 30);
      }
      return p;
    }
    
    template<template<class...> class Vec, class T>
    struct Workspace {
      
      Vec<T> x;
      Vec<T> x_old;
      Vec<T> x_pred;
      Vec<T> x_diff;      
      Vec<T> X_diff;
      Vec<T> f;
      Vec<T> f_old;
      Vec<T> f_diff;
      Vec<T> F_diff;
      Vec<T> gammas;
      
      int N;
      int memory;
      
      Workspace(int N, int memory):
        N(N),
        memory(memory),
        x(N),
        x_old(N),
        x_pred(N),
        x_diff(N),
        X_diff(N*memory),
        f(N),
        f_old(N),
        f_diff(N),
        F_diff(N*memory),
        gammas(memory){
        thrust::fill(X_diff.begin(), X_diff.end(), T());
        thrust::fill(F_diff.begin(), F_diff.end(), T());
      }        
    };

    template<template<class...> class Vec>
    inline  void updateMemoryVectors(Workspace<Vec, complex>& work,
                                     int iteration,
                                     const cudaStream_t st){
      
      
      int memory = work.memory;
      int N      = work.N;

      auto& f      = work.f;
      auto& f_old  = work.f_old;
      auto& f_diff = work.f_diff;
      auto& F_diff = work.F_diff;
      
      auto& x      = work.x;
      auto& x_old  = work.x_old;
      auto& x_diff = work.x_diff;
      auto& X_diff = work.X_diff;
      
      substract(f, f_old, f_diff, st);
      substract(x, x_old, x_diff, st);

      int column = iteration%memory;
      
      writeColumn(f_diff, F_diff, N, column, st);
      writeColumn(x_diff, X_diff, N, column, st);
    }
    
    template<template<class...> class Vec>
    inline  real computeRelativeError(Workspace<Vec, complex>& work,
                                      cudaStream_t st) {
      
      substract(work.x_pred, work.x, work.x_diff, st);
      real diff_norm = norm(work.x_diff, st);
      real x_norm    = std::max(real(1.0), norm(work.x, st));
      return diff_norm/x_norm;
    }
    
    static __global__ void updateSolutionAnderson_D(complex* X_diff, complex* F_diff,
                                                    complex* x_old,  complex* f_old,
                                                    complex* gammas, complex* x,
                                                    complex damping, int memory,
                                                    int dim){
      
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= dim) return;
      complex x_new_i = x_old[i] + damping * f_old[i];
      for (int j = 0; j<memory; j++){
        x_new_i -= (damping*F_diff[i+j*dim] + X_diff[i+j*dim])*gammas[j];
      }
      x[i] = x_new_i;
    }
    
    struct PicardStepFunctor {
      real damping;
      
      __host__ __device__
      explicit PicardStepFunctor(real damping_) : damping(damping_) {}
      
      __host__ __device__
      complex operator()(const complex& x_val, const complex& f_val) const {
        return x_val + damping * f_val;
      }
    };

    // x_{k+1} = x_k + damping * f_k
    template<template<class...> class Vec>
    inline void performPiccardStep(Workspace<Vec, complex>& work,
                                   real damping,
                                   cudaStream_t& st) {
      thrust::transform(thrust::cuda::par.on(st),
                        work.x_old.begin(), work.x_old.end(),
                        work.f_old.begin(),
                        work.x.begin(),
                        PicardStepFunctor(damping));
    }
    
    
    //x_{k+1} = x_k + damping*f - (X + damping * F)*gammas
    template<template<class...> class Vec>
    inline  void performAndersonStep(Workspace<Vec, complex>& work,
                                     LSWorkspace<Vec>& lswork,
                                     real damping,
                                     int niter,
                                     cudaStream_t &st){
      
      
      
      int memory   = work.memory;
      int cols     = std::min(memory, niter + 1);
      int rows     = work.N;
      lswork.cols  = cols;

      auto& F_diff = work.F_diff;
      auto& f_old  = work.f_old;
      auto& gammas = work.gammas;
      
      auto& X_diff = work.X_diff;
      auto& x_old  = work.x_old;
      auto& x      = work.x;
      
      if (memory == 0) {
        throw std::invalid_argument("[AAJ]: memory == 0 is invalid for Anderson acceleration");
      }
      
      if (cols <= 0 || cols > rows) {
        throw std::invalid_argument("[AAJ]: Invalid number of columns for least squares (cols <= 0 or cols > rows");
      }
      
      solve_least_squares(F_diff, f_old, gammas,
                          lswork, st);
      
      //if (!validGammas(gammas)) return performPiccardStep(x_old, f, damping, st);


      
      auto x_ptr      = thrust::raw_pointer_cast(x.data());
      auto f_old_ptr  = thrust::raw_pointer_cast(f_old.data());
      auto x_old_ptr  = thrust::raw_pointer_cast(x_old.data());
      auto X_diff_ptr = thrust::raw_pointer_cast(X_diff.data());
      auto F_diff_ptr = thrust::raw_pointer_cast(F_diff.data());
      auto gammas_ptr = thrust::raw_pointer_cast(gammas.data());

      int THREADS_PER_BLOCK = 128;
      int numBlocks = rows / THREADS_PER_BLOCK + 1;
      updateSolutionAnderson_D<<<numBlocks, THREADS_PER_BLOCK, 0, st>>>(X_diff_ptr, F_diff_ptr,
                                                                        x_old_ptr, f_old_ptr,
                                                                        gammas_ptr, x_ptr,
                                                                        damping, memory,
                                                                        rows);
    }
    
    template<template<class...> class Vec>
    inline void performNextStep(Workspace<Vec, complex>& work,
                                LSWorkspace<Vec>& lswork,
                                const Parameters &params,
                                int niter,
                                cudaStream_t &st){
      
      if (niter%params.notAcceleratedInterval == 0 and niter > 0){
        performAndersonStep(work, lswork, params.damping, niter, st);
        //CudaCheckError();
        
      } else {
        performPiccardStep(work, params.damping, st);
        //CudaCheckError();
      }
    }
    
    inline void handleErrorAndUpdate(real &error, const real &newerror,
                                     Parameters& aaj) {
      
    static int iterationsIncreasingError = 0;
    if (newerror > error || std::isnan(newerror) || std::isinf(newerror)) {
      iterationsIncreasingError++;
      if (iterationsIncreasingError == 5) {
        LOG_WARN("[AAJ] Anderson acceleration is not converging, trying to use a smaller damping");
        iterationsIncreasingError = 0;
        aaj.damping = aaj.damping*0.75;
      }
    }
    error = newerror;
    }
    
    inline  void printInfo(int printSteps, int niter, float error, bool verbose){
      if (niter%(printSteps) == 0 and niter > 0 and verbose){
        LOG_INFO("[AAJ] Current step: "<<niter);
        LOG_INFO("[AAJ] Current relative error: "<< error);
      }
    }

    template<template<class...> class Vec>
    inline void updateErrorAndLogging(Workspace<Vec, complex>& work,
                                      real& error,
                                      const Parameters& params,
                                      int currentIter,
                                      int totalIter, cudaStream_t st){
      
      if (currentIter % work.memory == 0 ||
        currentIter == params.maxIterations - 1) {
      real newerror = computeRelativeError(work, st);
      handleErrorAndUpdate(error, newerror, params);
    }
      printInfo(params.memory, totalIter, error, params.verbose);
    }
    
  // inline void printIterationOutcome(const real error, const real tolerance, int niter) {
  //   if (error > tolerance) {
  //     System::log<System::ERROR>("[AAJ] Iteration has reached the maximum number of steps without reaching the convergence.");
  //   } else {
  //     System::log<System::MESSAGE>("[AAJ] Iterative algorithm has successfully converged in: %i steps", niter);
  //   }
  // }
  }

  //Solves op(x)=x
  template<class Operator,template<class...> class Vec, class T>
  Result<Vec, T> solve(const Operator &op,
                       const Vec<T> &initialGuess,
                       const Parameters &params_in,
                       cudaStream_t st){
    
    int N             = initialGuess.size();
    Parameters params = detail::setDefaultMemory(params_in, N);
    real tolerance    = params.tolerance;
    int memory        = params.memory;
    int maxIterations = params.maxIterations;
    
    
    int totalNiter   = 0;
    int currentNiter = 0;
    real error       = params.tolerance + 1;
    
    LSWorkspace<Vec> lswork(N, memory);
    detail::Workspace<Vec, T> work(N, memory);
    
    auto& x      = work.x;
    auto& x_old  = work.x_old;
    auto& x_pred = work.x_pred;
    auto& f      = work.f;
    auto& f_old  = work.f_old;
    auto& X_diff = work.X_diff;
    auto& F_diff = work.F_diff;
    
    thrust::copy(thrust::cuda::par.on(st),
                 initialGuess.begin(),
                 initialGuess.end(),
                 x_old.begin());
    
    op(x_old, x, st);
    substract(x, x_old, f_old, st);
    op(x, x_pred, st);    
    
    //Run the AAJ loop
    while(error>tolerance && totalNiter<maxIterations){
      
      substract(x_pred, x, f, st);
      detail::updateMemoryVectors(work, currentNiter, st);
      x.swap(x_old);
      f.swap(f_old);
      detail::performNextStep(work, lswork, params, currentNiter, st);
      op(x, x_pred, st);
      detail::updateErrorAndLogging(work, error, params,
                                    currentNiter, totalNiter,
                                    st);
      
      ++totalNiter;
      ++currentNiter;
    }
    
    //detail::printIterationOutcome(error, params.tolerance, totalNiter);
    // params.employedIterations = totalNiter;
    // params.finalError         = error;
    // params.damping            = damping_0;
    
    Result<Vec, T> result{x, {}};
    return result;
  }
}
