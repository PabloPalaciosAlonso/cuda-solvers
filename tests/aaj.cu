#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include "cuda_solvers/types.h"
#include "cuda_solvers/aaj.h"
#include "cuda_solvers/detail/least_squares.cuh"

using namespace cuda_solvers;
using namespace aaj;

TEST(AAJ, updateMemoryVector_firstIteration) {
  cudaStream_t st = 0;
  const int N      = 3;
  const int memory = 2;
  const int iter   = 0;
  detail::Workspace<thrust::device_vector, complex> work(N, memory);

  work.x_old = {{1,1},{2,1},{-1,2}};
  work.x     = {{-0.5,-0.5},{-2,3},{1,-2}};

  work.f_old = {{2,11},{12,13},{-41,24}};
  work.f     = {{-10.5,-0.15},{-22,33},{31,-23}};
  
  detail::updateMemoryVectors(work, iter, st);
  
  for (int i = 0; i < N; ++i) {
    complex x_expected = complex(work.x[i]) - complex(work.x_old[i]);
    complex f_expected = complex(work.f[i]) - complex(work.f_old[i]);
    EXPECT_EQ(complex(work.X_diff[i]).real(), x_expected.real());
    EXPECT_EQ(complex(work.X_diff[i]).imag(), x_expected.imag());
    EXPECT_EQ(complex(work.F_diff[i]).real(), f_expected.real());
    EXPECT_EQ(complex(work.F_diff[i]).imag(), f_expected.imag());    
  }
}

TEST(AAJ, updateMemoryVector_ArbitraryIteration) {
  cudaStream_t st  = 0;
  const int N      = 4;
  const int memory = 3;
  const int iter   = 2;
  detail::Workspace<thrust::device_vector, complex> work(N, memory);

  work.x_old = {{1,1},{2,1}, {-1,2}, {-2,2}};
  work.x     = {{-0.5,-0.5}, {-2,3}, {1,-2}, {0.5,-2.5}};

  work.f_old = {{2,11},{12,13},{-41,24}, {-0.25,0.5}};
  work.f     = {{-10.5,-0.15},{-22,33},{31,-23}, {-0.15,-5}};
  
  detail::updateMemoryVectors(work, iter, st);
  
  for (int i = 0; i < N; ++i) {
    complex x_expected = complex(work.x[i]) - complex(work.x_old[i]);
    complex f_expected = complex(work.f[i]) - complex(work.f_old[i]);

    EXPECT_EQ(complex(work.X_diff[index2D(i,iter, N)]).real(), x_expected.real());
    EXPECT_EQ(complex(work.X_diff[index2D(i,iter, N)]).imag(), x_expected.imag());
    EXPECT_EQ(complex(work.F_diff[index2D(i,iter, N)]).real(), f_expected.real());
    EXPECT_EQ(complex(work.F_diff[index2D(i,iter, N)]).imag(), f_expected.imag());    
  }
}

TEST(AAJ, piccardIterationNoDamping) {
  cudaStream_t st  = 0;
  const int N      = 4;
  const int memory = 3;

  detail::Workspace<thrust::device_vector, complex> work(N, memory);

  work.x_old = {{1,1}, {2,1}, {-1,2}, {-2,2}};
  work.f_old = {{-10.5,-0.15}, {-22,33}, {31,-23}, {-0.15,-5}};

  real damping = real(1.0);

  detail::performPiccardStep(work, damping, st);

  for (int i = 0; i < N; ++i) {
    complex target = complex(work.x_old[i]) + complex(work.f_old[i]);

    EXPECT_EQ(target.real(), complex(work.x[i]).real());
    EXPECT_EQ(target.imag(), complex(work.x[i]).imag());
  }
}

TEST(AAJ, piccardIteration){
  cudaStream_t st  = 0;
  const int N      = 4;
  const int memory = 3;
  detail::Workspace<thrust::device_vector, complex> work(N, memory);
  work.x_old = {{1,1},{2,1}, {-1,2}, {-2,2}};
  work.f_old = {{-10.5,-0.15},{-22,33},{31,-23}, {-0.15,-5}};

  real damping = 0.3;

  detail::performPiccardStep(work, damping, st);
  for (int i = 0; i<N; i++){
    complex target = complex(work.x_old[i]) + damping * complex(work.f_old[i]);
    EXPECT_EQ(target.real(), complex(work.x[i]).real());
    EXPECT_EQ(target.imag(), complex(work.x[i]).imag());
  }
}

TEST(AAJ, andersonStepExactResidualColumnReturnsOldX) {
  cudaStream_t st  = 0;
  const int N      = 4;
  const int memory = 1;
  const int niter  = 0;

  detail::Workspace<thrust::device_vector, complex> work(N, memory);
  LSWorkspace lswork(N, memory);

  work.x_old = {{1,1}, {2,1}, {-1,2}, {-2,2}};
  work.f_old = {{0.5,-0.25}, {-1,2}, {3,-1}, {-0.5,0.75}};

  thrust::fill(work.X_diff.begin(), work.X_diff.end(), complex(0,0));
  thrust::fill(work.F_diff.begin(), work.F_diff.end(), complex(0,0));
  thrust::fill(work.gammas.begin(), work.gammas.end(), complex(0,0));

  for (int i = 0; i < N; ++i) {
    work.F_diff[index2D(i, 0, N)] = work.f_old[i];
    work.X_diff[index2D(i, 0, N)] = complex(0,0);
  }

  real damping = real(0.3);

  detail::performAndersonStep(work, lswork, damping, niter, st);
  cudaStreamSynchronize(st);

  for (int i = 0; i < N; ++i) {
    complex target = complex(work.x_old[i]);

    EXPECT_NEAR(target.real(), complex(work.x[i]).real(), 1e-10);
    EXPECT_NEAR(target.imag(), complex(work.x[i]).imag(), 1e-10);
  }
}

TEST(AAJ, andersonStepExactResidualColumnIncludesXDiffCorrection) {
  cudaStream_t st  = 0;
  const int N      = 4;
  const int memory = 1;
  const int niter  = 0;

  detail::Workspace<thrust::device_vector, complex> work(N, memory);
  LSWorkspace lswork(N, memory);

  work.x_old = {{1,1}, {2,1}, {-1,2}, {-2,2}};
  work.f_old = {{0.5,-0.25}, {-1,2}, {3,-1}, {-0.5,0.75}};

  thrust::device_vector<complex> d = {
    {0.1, 0.2},
    {-0.3, 0.5},
    {1.0, -0.25},
    {-0.5, -0.75}
  };

  thrust::fill(work.X_diff.begin(), work.X_diff.end(), complex(0,0));
  thrust::fill(work.F_diff.begin(), work.F_diff.end(), complex(0,0));
  thrust::fill(work.gammas.begin(), work.gammas.end(), complex(0,0));

  for (int i = 0; i < N; ++i) {
    work.F_diff[index2D(i, 0, N)] = work.f_old[i];
    work.X_diff[index2D(i, 0, N)] = d[i];
  }

  real damping = real(0.3);

  detail::performAndersonStep(work, lswork, damping, niter, st);
  cudaStreamSynchronize(st);

  for (int i = 0; i < N; ++i) {
    complex target = complex(work.x_old[i]) - complex(d[i]);

    EXPECT_NEAR(target.real(), complex(work.x[i]).real(), 1e-10);
    EXPECT_NEAR(target.imag(), complex(work.x[i]).imag(), 1e-10);
  }
}

TEST(AAJ, AAJComplexFunction) {
  cudaStream_t st = 0;
  
  const int N = 5;
  
  thrust::device_vector<complex> x0 = {
    {0.9, 0.2},
    {0.9, 0.1},
    {0.9, 0.5},
    {0.9, 0.8},
    {0.9, 0.2}
  };

  struct ExpImFixedPointOp{
    void operator()(const thrust::device_vector<complex>& x,
                    thrust::device_vector<complex>& out,
                    cudaStream_t st) const {
      for (int i = 1; i <= x.size(); ++i) {
        complex xi = x[i-1];
        out[i-1] = exp(xi*complex(1,1))/i;
      }
    }
  };
  
  
  Parameters p;
  p.memory                 = 4;
  p.maxIterations          = 100;
  p.damping                = 0.3;
  p.tolerance              = 1e-7;
  p.notAcceleratedInterval = 2;
  
  ExpImFixedPointOp op;
  
  auto result = solve(op, x0, p, st);
  auto x      = result.x;
  
  thrust::device_vector<complex> opx(x.size());
  op(x, opx, st);
  
  for (int i = 0; i < N; ++i) {
    complex xi   = complex(x[i]);
    complex opxi = complex(opx[i]);
    
    EXPECT_NEAR(xi.real(), opxi.real(), p.tolerance);
    EXPECT_NEAR(xi.imag(), opxi.imag(), p.tolerance);
  }
  //EXPECT_LE(p.employedIterations, p.maxIterations);
}
