#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include "cuda_solvers/types.h"
#include "cuda_solvers/detail/least_squares.cuh"
#include "cuda_solvers/detail/utils.h"

using namespace cuda_solvers;

inline void expectComplexNear(const complex& a, const complex& b, real tol) {
  EXPECT_NEAR(complex(a).real(), complex(b).real(), tol);
  EXPECT_NEAR(complex(a).imag(), complex(b).imag(), tol);
}

#ifdef DOUBLE_PRECISION
constexpr real LS_TOL = real(1e-10);
#else
constexpr real LS_TOL = real(1e-4);
#endif

TEST(LeastSquares, identitySquareSystem) {
  cudaStream_t st = 0;
  
  const int rows = 3;
  const int cols = 3;
  
  thrust::device_vector<complex> A(rows * cols, complex(0, 0));
  thrust::device_vector<complex> b = {
    {1, 2},
    {-3, 0.5},
    {0.25, -1}
  };

  thrust::device_vector<complex> x(cols, complex(0, 0));

  for (int i = 0; i < rows; ++i) {
    A[index2D(i, i, rows)] = complex(1, 0);
  }

  LSWorkspace<thrust::device_vector> ws(rows, cols);

  solve_least_squares(A, b, x, ws, st);
  cudaStreamSynchronize(st);

  for (int i = 0; i < cols; ++i) {
    expectComplexNear(x[i], b[i], LS_TOL);
  }
}

TEST(LeastSquares, overdeterminedExactSystem) {
  cudaStream_t st = 0;

  const int rows = 4;
  const int cols = 2;

  thrust::device_vector<complex> A(rows * cols, complex(0, 0));
  thrust::device_vector<complex> b(rows, complex(0, 0));
  thrust::device_vector<complex> x(cols, complex(0, 0));

  // A =
  // [1 0]
  // [0 1]
  // [1 1]
  // [2 -1]
  A[index2D(0, 0, rows)] = complex(1, 0);
  A[index2D(1, 0, rows)] = complex(0, 0);
  A[index2D(2, 0, rows)] = complex(1, 0);
  A[index2D(3, 0, rows)] = complex(2, 0);

  A[index2D(0, 1, rows)] = complex(0, 0);
  A[index2D(1, 1, rows)] = complex(1, 0);
  A[index2D(2, 1, rows)] = complex(1, 0);
  A[index2D(3, 1, rows)] = complex(-1, 0);

  complex x0 = complex(2, -1);
  complex x1 = complex(-0.5, 0.25);

  for (int i = 0; i < rows; ++i) {
    b[i] = complex(A[index2D(i, 0, rows)]) * x0 +
      complex(A[index2D(i, 1, rows)]) * x1;
  }

  LSWorkspace<thrust::device_vector> ws(rows, cols);

  solve_least_squares(A, b, x, ws, st);
  cudaStreamSynchronize(st);

  expectComplexNear(x[0], x0, LS_TOL);
  expectComplexNear(x[1], x1, LS_TOL);
}

TEST(LeastSquares, overdeterminedOneColumnMeanSolution) {
  cudaStream_t st = 0;

  const int rows = 3;
  const int cols = 1;

  thrust::device_vector<complex> A(rows * cols, complex(1, 0));

  thrust::device_vector<complex> b = {
    {1, 0},
    {2, 0},
    {4, 0}
  };

  thrust::device_vector<complex> x(cols, complex(0, 0));

  LSWorkspace<thrust::device_vector> ws(rows, cols);

  solve_least_squares(A, b, x, ws, st);
  cudaStreamSynchronize(st);

  complex expected = complex(real(7.0) / real(3.0), 0);

  expectComplexNear(x[0], expected, LS_TOL);
}

TEST(LeastSquares, doesNotModifyInputAOrB) {
  cudaStream_t st = 0;

  const int rows = 4;
  const int cols = 2;

  thrust::device_vector<complex> A(rows * cols, complex(0, 0));
  thrust::device_vector<complex> b(rows, complex(0, 0));
  thrust::device_vector<complex> x(cols, complex(0, 0));

  A[index2D(0, 0, rows)] = complex(1, 0);
  A[index2D(1, 0, rows)] = complex(0, 0);
  A[index2D(2, 0, rows)] = complex(1, 0);
  A[index2D(3, 0, rows)] = complex(2, 0);

  A[index2D(0, 1, rows)] = complex(0, 0);
  A[index2D(1, 1, rows)] = complex(1, 0);
  A[index2D(2, 1, rows)] = complex(1, 0);
  A[index2D(3, 1, rows)] = complex(-1, 0);

  b = {
    {1, 0},
    {2, 0},
    {3, 0},
    {0, 0}
  };

  thrust::device_vector<complex> A_before = A;
  thrust::device_vector<complex> b_before = b;

  LSWorkspace<thrust::device_vector> ws(rows, cols);

  solve_least_squares(A, b, x, ws, st);
  cudaStreamSynchronize(st);

  for (int i = 0; i < rows * cols; ++i) {
    expectComplexNear(A[i], A_before[i], LS_TOL);
  }

  for (int i = 0; i < rows; ++i) {
    expectComplexNear(b[i], b_before[i], LS_TOL);
  }
}

TEST(LeastSquares, devInfoIsZeroForValidProblem) {
  cudaStream_t st = 0;

  const int rows = 3;
  const int cols = 1;

  thrust::device_vector<complex> A(rows * cols, complex(1, 0));
  thrust::device_vector<complex> b = {
    {1, 0},
    {2, 0},
    {3, 0}
  };
  thrust::device_vector<complex> x(cols, complex(0, 0));

  LSWorkspace<thrust::device_vector> ws(rows, cols);

  solve_least_squares(A, b, x, ws, st);
  cudaStreamSynchronize(st);

  int info = -1;
  cudaMemcpy(&info,
             thrust::raw_pointer_cast(ws.info.data()),
             sizeof(int),
             cudaMemcpyDeviceToHost);

  EXPECT_EQ(info, 0);
}
