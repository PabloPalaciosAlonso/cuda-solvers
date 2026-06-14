#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include "cuda_solvers/types.h"
#include "cuda_solvers/aaj.h"
#include "cuda_solvers/detail/least_squares.cuh"

using namespace cuda_solvers;
using namespace aaj;

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
  p.memory         = 4;
  p.maxIterations = 100;
  p.damping        = 0.3;
  p.tolerance      = 1e-7;
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
