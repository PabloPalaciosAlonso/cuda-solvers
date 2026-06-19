#include <thrust/execution_policy.h>
#include "cuda_solvers/types.h"
#include "cuda_solvers/gmres.h"
#include "cuda_solvers/detail/vector_operations.cuh"
#include "cuda_solvers/detail/utils.h"
#include "cuda_solvers/detail/logger.h"

namespace cuda_solvers::gmres{
  
  namespace detail{

    template<template<class...> class Vec, class T>
    struct Workspace {
      Vec<T> krylovBase;
      Vec<T> residue;
      Vec<T> Ax;
      Vec<T> v0;
      Vec<T> vi;
      Vec<T> vj;
      Vec<T> vk;
      Vec<T> w;

      std::vector<T> hessenberg;
      std::vector<real> cosGivens;
      std::vector<T> sinGivens;
      std::vector<T> g;
      std::vector<T> y;

      int N;
      int restart;
      
      Workspace(int N, int restart)
        : krylovBase(N * (restart + 1), T()),
          residue(N),
          Ax(N),
          v0(N),
          vk(N),
          vj(N),
          vi(N),
          w(N),
          hessenberg((restart + 1) * restart),
          cosGivens(restart),
          sinGivens(restart),
          g(restart + 1),
          y(restart),
          N(N),
          restart(restart){}
    };

    
    // Given the equation Ax = b; computes b-Ax for a given x; op(x,st) \def Ax 
    template<class Operator, template<class...> class Vec, class T>
    void computeResidue(const Operator& op,
                        const Vec<T>& x,
                        const Vec<T>& b,
                        Workspace<Vec, T>& work,
                        const cudaStream_t st){

      assert(work.N == b.size());
      op(x, work.Ax, st);
      substract(b, work.Ax, work.residue, st);
    }
    
    template<class Operator, template<class...> class Vec, class T>
    bool runArnoldiStep(const Operator& op,
                        Workspace<Vec, T> &work, int j,
                        cudaStream_t st) {
      
      auto& vi         = work.vi;
      auto& vj         = work.vj;
      auto& w          = work.w;
      auto& H          = work.hessenberg;
      auto& krylovBase = work.krylovBase;

      int N       = work.N;
      int restart = work.restart;
      
      getColumn(krylovBase, vj, j, st);
      op(vj, w, st);
      
      for (int i = 0; i < j + 1; ++i) {
        getColumn(krylovBase, vi, i, st);
        const T hij = dotc(vi, w, st);
        H[index2D(i, j, restart + 1)] = hij;
        axpy(vi, w, w, -hij, st);
      }
      
      const real hnext              = norm(w, st);
      H[index2D(j + 1, j, restart + 1)] = T(hnext);
      
      if (std::abs(hnext) <= std::numeric_limits<real>::epsilon()) {
        return false;
      }
      
      divide(w, T(hnext), w, st);
      writeColumn(w, krylovBase, N, j + 1, st);
      return true;
    }
    
    template<class T>
    inline std::vector<T> solveUpperTriangular(const std::vector<T>& R,
                                               const std::vector<T>& g) {
      int ncols = g.size();
      std::vector<T> y(ncols);
      
      for (int i = ncols - 1; i >= 0; --i) {
        T s = g[i];
        for (int j = i + 1; j < ncols; ++j) {
          s -= R[index2D(i, j, ncols)] * y[j];
        }
        
        const T diag = R[index2D(i, i, ncols)];
        dotProductFunctor<T> sq;
        real norm = sqrt(make_real(sq(diag, diag)));
        if (norm <= std::numeric_limits<real>::epsilon()) {
          y[i] = real(0);
        } else {
          y[i] = s / diag;
        }
      }
      return y;
    }
    
    inline std::pair<real, real> apply2Drotation(const real c,
                                                 const real s,
                                                 const real v1,
                                                 const real v2){
      
      real v1rot = c * v1 - s * v2;
      real v2rot = s * v1 + c * v2;
      return {v1rot, v2rot};
    }
    
    inline std::pair<complex, complex> apply2Drotation(const real c,
                                                       const complex s,
                                                       const complex v1,
                                                       const complex v2) {
      const complex v1rot = c * v1 - thrust::conj(s) * v2;
      const complex v2rot = s * v1 + c * v2;
      
      return {v1rot, v2rot};
    }
    
    inline std::pair<real, real> computeGivensRotation(const real h1,
                                                       const real h2) {
      const real denom = std::hypot(h1, h2);
      if (denom <= std::numeric_limits<real>::epsilon()) {
        return {real(1), real(0)};
      }
      
      return {h1 / denom, -h2 / denom};
    }
    
    inline std::pair<real, complex>
    computeGivensRotation(const complex h1,
                          const complex h2) {
      const real abs_h1 = thrust::abs(h1);
      const real abs_h2 = thrust::abs(h2);
      
      if (abs_h2 <= std::numeric_limits<real>::epsilon()) {
        return {real(1), complex(0)};
      }
      
      if (abs_h1 <= std::numeric_limits<real>::epsilon()) {
        return {real(0), -h2 / abs_h2};
      }
      
      const real denom = std::hypot(abs_h1, abs_h2);
      
      const real c = abs_h1 / denom;
      
      // Chosen so that:
      // s * h1 + c * h2 = 0
      const complex s = -h2 * thrust::conj(h1) / (denom * abs_h1);
      
      return {c, s};
    }

    template<template<class...> class Vec>
    inline real triangularizeGivens(Workspace<Vec, real>& work,
                                    const int j,
                                    const real norm_b) {
      
      auto& hessenberg = work.hessenberg;
      auto& cosGivens  = work.cosGivens;
      auto& sinGivens  = work.sinGivens;
      auto& g          = work.g;
      int restart      = work.restart;
      
      for (int i = 0; i < j + 1; ++i) {
        const real hij  = hessenberg[index2D(i,     j, restart + 1)];
        const real hi1j = hessenberg[index2D(i + 1, j, restart + 1)];
        real c_i = cosGivens[i];
        real s_i = sinGivens[i];
        
        if (i==j){
          std::tie(c_i, s_i) = computeGivensRotation(hij, hi1j);
          cosGivens[i] = c_i;
          sinGivens[i] = s_i;
        }
        
        const auto [hijrot, hi1jrot] = apply2Drotation(c_i, s_i,
                                                       hij, hi1j);
        
        hessenberg[index2D(i, j, restart + 1)]     = hijrot;
        hessenberg[index2D(i + 1, j, restart + 1)] = hi1jrot;
      }
      
      auto [newgj, newgj1] = apply2Drotation(cosGivens[j],
                                             sinGivens[j],
                                             g[j], g[j+1]);
      
      g[j]     = newgj;
      g[j + 1] = newgj1;
      
      return std::abs(g[j + 1]) / norm_b;
    }

    template<template<class...> class Vec>
    inline real triangularizeGivens(Workspace<Vec, complex>& work,
                                    const int j,
                                    const real norm_b) {
      
      auto& hessenberg = work.hessenberg;
      auto& cosGivens  = work.cosGivens;
      auto& sinGivens  = work.sinGivens;
      auto& g          = work.g;
      int restart      = work.restart;
      
      for (int i = 0; i < j + 1; ++i) {
        const complex hij  = hessenberg[index2D(i,     j, restart + 1)];
        const complex hi1j = hessenberg[index2D(i + 1, j, restart + 1)];
        
        real c_i    = cosGivens[i];
        complex s_i = sinGivens[i];
    
        if (i == j) {
          std::tie(c_i, s_i) = computeGivensRotation(hij, hi1j);
          cosGivens[i] = c_i;
          sinGivens[i] = s_i;
        }
    
        const auto [hijrot, hi1jrot] =
          apply2Drotation(c_i, s_i, hij, hi1j);
    
        hessenberg[index2D(i,     j, restart + 1)] = hijrot;
        hessenberg[index2D(i + 1, j, restart + 1)] = hi1jrot;
      }
  
      auto [newgj, newgj1] =
        apply2Drotation(cosGivens[j],
                        sinGivens[j],
                        g[j],
                        g[j + 1]);
  
      g[j]     = newgj;
      g[j + 1] = newgj1;
  
      return thrust::abs(g[j + 1]) / norm_b;
    }
  }
  
  // Solves op(x, st) = b; with op(x,st) \def A*x being A a matrix
  template<class Operator,template<class...> class Vec, class T>
  Result<Vec, T> solve(const Operator& op,
                       const Vec<T>& b,
                       const Vec<T>& initialGuess,
                       const Parameters& params,
                       cudaStream_t st) {
    
    const int N = b.size();
    assert(initialGuess.size() == b.size());
    assert(params.memory > 0);
    
    Result<Vec, T> result{initialGuess, {}};
    auto& x    = result.x;
    auto& info = result.info;
    
    const int restart = params.memory;
    const real tol    = params.tolerance;   
    const real norm_b = norm(b, st);

    if (norm_b == real(0)) {
      x    = Vec<T>(N, T());
      info = writeInfo(true, 0,0.0, params.verbose, "GMRES");
      return result;
    }
    
    detail::Workspace<Vec, T> work(N, restart);
    int total_iters = 0;
    
    for (int outer = 0; outer < params.maxIterations; ++outer) {
      
      detail::computeResidue(op, x, b, work, st);
      real beta                 = norm(work.residue, st);
      real relres               = beta / norm_b;

      if (params.verbose){
        LOG_INFO("[GMRES] Current step: " << outer * restart);
        LOG_INFO("[GMRES] Current relative error: " << relres);
      }
      if (relres < tol) {
        info = writeInfo(true, outer * restart, relres, params.verbose, "GMRES");
        return result;
      }
      
      std::fill(work.hessenberg.begin(), work.hessenberg.end(), T());
      std::fill(work.cosGivens.begin(), work.cosGivens.end(), real(0));
      std::fill(work.sinGivens.begin(), work.sinGivens.end(), T());
      std::fill(work.g.begin(), work.g.end(), T());
      
      // v0 = residue / beta
      divide(work.residue, T(beta), work.v0, st);
      writeColumn(work.v0, work.krylovBase, N, 0, st);
      work.g[0] = beta;
      
      int j_final = restart;
      for (int j = 0; j < restart; ++j) {
        total_iters++;
        
        const bool hasNextKrylovVector = detail::runArnoldiStep(op, work, j, st);
        relres = detail::triangularizeGivens(work, j, norm_b);
        
        if (relres < tol || !hasNextKrylovVector) {
          j_final = j + 1;
          break;
        }
      }
      
      std::vector<T> Rsmall(j_final * j_final, T());
      std::vector<T> gsmall(j_final, T());
      
      for (int i = 0; i < j_final; ++i) {
        gsmall[i] = work.g[i];
        for (int j = 0; j < j_final; ++j) {
          Rsmall[index2D(i, j, j_final)] = work.hessenberg[index2D(i, j, restart + 1)];
        }
      }
      
      std::vector<T> y = detail::solveUpperTriangular(Rsmall, gsmall);
      
      for (int k = 0; k < j_final; ++k) {
        getColumn(work.krylovBase, work.vk, k, st);
        axpy(work.vk, x, x, y[k], st);
      }
      
      detail::computeResidue(op, x, b, work, st);
      beta     = norm(work.residue, st);
      relres   = beta / norm_b;
      
      if (relres < tol) {
        info = writeInfo(true, total_iters, relres, params.verbose, "GMRES");
        return result;
      }     
    }

    detail::computeResidue(op, x, b, work, st);
    const real beta           = norm(work.residue, st);
    const real relres         = beta / norm_b;
    
    info = writeInfo(false, total_iters, relres, params.verbose, "GMRES");
    return result;
  } 
}
