#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include "cuda_solvers/types.h"
#include "cuda_solvers/gmres.h"

using namespace cuda_solvers;
using namespace gmres;

TEST(GMRES, computeResidueReal) {
  cudaStream_t st = 0;
  const int N = 3;
  
  std::vector<real> x_h = {
    real(1.0),
    real(2.0),
    real(-1.0)
  };

  std::vector<real> b_h = {
    real(5.0),
    real(1.0),
    real(4.0)
  };

  std::vector<real> A = {
    real(2.0), real(0.0), real(1.0),
    real(0.0), real(-1.0), real(3.0),
    real(4.0), real(1.0), real(0.0)
  };

  thrust::device_vector<real> x(x_h);
  thrust::device_vector<real> b(b_h);

  struct Operator {
    std::vector<real> A;
    int N;

    void operator()(const thrust::device_vector<real>& x,
                    thrust::device_vector<real>& Ax,
                    cudaStream_t st) const {
      (void) st;

      thrust::host_vector<real> hx = x;
      thrust::host_vector<real> hAx(N, real(0));

      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          hAx[i] += A[i * N + j] * hx[j];
        }
      }

      Ax = hAx;
    }
  };

  Operator op{A, N};

  detail::Workspace<real> work(N, 10);
  detail::computeResidue(op, x, b, work, st);

  std::vector<real> expected(N, real(0));
  for (int i = 0; i < N; ++i) {
    real Axi = real(0);
    for (int j = 0; j < N; ++j) {
      Axi += A[i * N + j] * x_h[j];
    }
    expected[i] = b_h[i] - Axi;
  }

  thrust::host_vector<real> residue_h = work.residue;

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(residue_h[i], expected[i], 1e-12);
  }
}

TEST(GMRES, computeResidueComplex) {
  cudaStream_t st = 0;
  const int N = 3;

  std::vector<complex> x_h = {
    complex( 1.0,  2.0),
    complex( 2.0, -1.0),
    complex(-1.0,  0.5)
  };

  std::vector<complex> b_h = {
    complex( 5.0, -1.0),
    complex( 1.0,  3.0),
    complex( 4.0, -2.0)
  };

  std::vector<complex> A = {
    complex(2.0,  1.0), complex( 0.0, 0.0), complex(1.0, -1.0),
    complex(0.0,  0.0), complex(-1.0, 0.0), complex(3.0,  2.0),
    complex(4.0, -1.0), complex( 1.0, 0.0), complex(0.0,  0.0)
  };

  thrust::device_vector<complex> x(x_h);
  thrust::device_vector<complex> b(b_h);

  struct Operator {
    std::vector<complex> A;
    int N;

    void operator()(const thrust::device_vector<complex>& x,
                    thrust::device_vector<complex>& Ax,
                    cudaStream_t st) const {
      (void) st;

      thrust::host_vector<complex> hx = x;
      thrust::host_vector<complex> hAx(N, complex(0.0, 0.0));

      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          hAx[i] += A[i * N + j] * hx[j];
        }
      }

      Ax = hAx;
    }
  };

  Operator op{A, N};

  detail::Workspace<complex> work(N, 10);
  detail::computeResidue(op, x, b, work, st);

  std::vector<complex> expected(N, complex(0.0, 0.0));

  for (int i = 0; i < N; ++i) {
    complex Axi = complex(0.0, 0.0);

    for (int j = 0; j < N; ++j) {
      Axi += A[i * N + j] * x_h[j];
    }

    expected[i] = b_h[i] - Axi;
  }

  thrust::host_vector<complex> residue_h = work.residue;

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(residue_h[i].real(), expected[i].real(), 1e-12);
    EXPECT_NEAR(residue_h[i].imag(), expected[i].imag(), 1e-12);
  }
}

TEST(GMRES, apply2Drotation){

  real phi = M_PI/3; //60º
  real r   = 3.5;
  real x   = r*cos(phi);
  real y   = r*sin(phi);

  real rotAng = M_PI/8; //22.5º
  real cang   = cos(rotAng);
  real sang   = sin(rotAng);

  auto [newX, newY] = detail::apply2Drotation(cang, sang, x, y);

  real newr = std::hypot(newX, newY);
  real targetX = r * (cos(phi+rotAng));
  real targetY = r * (sin(phi+rotAng));
  
  EXPECT_NEAR(r, newr, 1e-8);
  EXPECT_NEAR(newX, targetX, 1e-8);
  EXPECT_NEAR(newY, targetY, 1e-8);   
}

TEST(GMRES, computeGivensRotation){
  real tol = 1e-8;
  real h1  = 3.0;
  real h2  = 5.0;
  
  auto [c, s]         = detail::computeGivensRotation(h1, h2);
  auto [h1rot, h2rot] = detail::apply2Drotation(c, s, h1, h2);
  
  EXPECT_NEAR(c*c+s*s, 1.0, tol);
  EXPECT_NEAR(h1rot, sqrt(h1*h1+h2*h2), tol);
  EXPECT_NEAR(h2rot, 0.0, tol);
}

TEST(GMRES, triangularizeGivensReal) {
  const int restart = 5;
  const real norm_b = real(2.1);

  detail::Workspace<real> work(1, restart);

  std::fill(work.hessenberg.begin(), work.hessenberg.end(), real(0));
  std::fill(work.cosGivens.begin(),  work.cosGivens.end(),  real(0));
  std::fill(work.sinGivens.begin(),  work.sinGivens.end(),  real(0));
  std::fill(work.g.begin(),          work.g.end(),          real(0));

  for (int j = 0; j < restart; ++j) {
    for (int i = 0; i <= j + 1; ++i) {
      work.hessenberg[index2D(i, j, restart)] =
          static_cast<real>(1 + i + 3 * j);
    }
  }

  work.g[0] = norm_b;

  for (int j = 0; j < restart; ++j) {
    detail::triangularizeGivens(work, j, norm_b);

    EXPECT_NEAR(work.hessenberg[index2D(j + 1, j, restart)], real(0), 1e-12);
  }

  for (int j = 0; j < restart; ++j) {
    for (int i = j + 1; i < restart + 1; ++i) {
      EXPECT_NEAR(work.hessenberg[index2D(i, j, restart)], real(0), 1e-12);
    }
  }
}


TEST(GMRES, apply2DrotationComplex) {
  const real tol = 1e-8;

  const complex v1 = complex(1.2, -0.7);
  const complex v2 = complex(2.3,  1.5);

  const real rotAng = M_PI / 8.0;
  const real c = std::cos(rotAng);
  const complex s = complex(std::sin(rotAng), 0.0);

  auto [newV1, newV2] = detail::apply2Drotation(c, s, v1, v2);

  const complex targetV1 = c * v1 - thrust::conj(s) * v2;
  const complex targetV2 = s * v1 + c * v2;

  EXPECT_NEAR(thrust::abs(newV1 - targetV1), 0.0, tol);
  EXPECT_NEAR(thrust::abs(newV2 - targetV2), 0.0, tol);

  const real oldNorm = std::sqrt(thrust::norm(v1) + thrust::norm(v2));
  const real newNorm = std::sqrt(thrust::norm(newV1) + thrust::norm(newV2));

  EXPECT_NEAR(newNorm, oldNorm, tol);
}

TEST(GMRES, computeGivensRotationComplex) {
  const real tol = 1e-8;

  const complex h1 = complex(3.0, -2.0);
  const complex h2 = complex(5.0,  4.0);

  auto [c, s] = detail::computeGivensRotation(h1, h2);
  auto [h1rot, h2rot] = detail::apply2Drotation(c, s, h1, h2);

  EXPECT_NEAR(c * c + thrust::norm(s), 1.0, tol);
  EXPECT_NEAR(thrust::abs(h2rot), 0.0, tol);

  const real targetNorm = std::sqrt(thrust::norm(h1) + thrust::norm(h2));
  EXPECT_NEAR(thrust::abs(h1rot), targetNorm, tol);
}

TEST(GMRES, triangularizeGivensComplex) {
  const int restart = 5;
  const real norm_b = real(2.1);
  const real tol    = real(1e-12);

  detail::Workspace<complex> work(1, restart);

  std::fill(work.hessenberg.begin(), work.hessenberg.end(), complex(0.0, 0.0));
  std::fill(work.cosGivens.begin(),  work.cosGivens.end(),  real(0));
  std::fill(work.sinGivens.begin(),  work.sinGivens.end(),  complex(0.0, 0.0));
  std::fill(work.g.begin(),          work.g.end(),          complex(0.0, 0.0));

  // Hessenberg (restart + 1) x restart
  for (int j = 0; j < restart; ++j) {
    for (int i = 0; i <= j + 1; ++i) {
      const real re = static_cast<real>(1 + i + 3 * j);
      const real im = static_cast<real>(0.2 * (i + 1) - 0.1 * (j + 1));

      work.hessenberg[index2D(i, j, restart)] = complex(re, im);
    }
  }

  work.g[0] = complex(norm_b, 0.0);

  for (int j = 0; j < restart; ++j) {
    detail::triangularizeGivens(work, j, norm_b);

    EXPECT_NEAR(thrust::abs(work.hessenberg[index2D(j + 1, j, restart)]),
                real(0),
                tol);
  }

  for (int j = 0; j < restart; ++j) {
    for (int i = j + 1; i < restart + 1; ++i) {
      EXPECT_NEAR(thrust::abs(work.hessenberg[index2D(i, j, restart)]),
                  real(0),
                  tol);
    }
  }
}

TEST(GMRES, apply2DrotationComplexPreservesNormWithComplexS) {
  const real tol = real(1e-8);

  const complex v1 = complex(1.0,  2.0);
  const complex v2 = complex(3.0, -1.0);

  const real c = real(0.8);
  const complex s = complex(
    real(0.3),
    -std::sqrt(real(1.0) - c * c - real(0.3) * real(0.3))
  );

  ASSERT_NEAR(c * c + thrust::norm(s), real(1.0), tol);

  auto [newV1, newV2] = detail::apply2Drotation(c, s, v1, v2);

  const real oldNorm = std::sqrt(thrust::norm(v1) + thrust::norm(v2));
  const real newNorm = std::sqrt(thrust::norm(newV1) + thrust::norm(newV2));

  EXPECT_NEAR(newNorm, oldNorm, tol);
}

TEST(GMRES, solveUpperTriangularReal) {
  const int n = 4;
  
  std::vector<real> R(n * n, real(0));
  
  R[index2D(0, 0, n)] = real( 2);
  R[index2D(0, 1, n)] = real(-1);
  R[index2D(0, 2, n)] = real( 3);
  R[index2D(0, 3, n)] = real( 1);

  R[index2D(1, 1, n)] = real( 4);
  R[index2D(1, 2, n)] = real(-2);
  R[index2D(1, 3, n)] = real( 2);

  R[index2D(2, 2, n)] = real( 5);
  R[index2D(2, 3, n)] = real(-3);

  R[index2D(3, 3, n)] = real(-2);

  const std::vector<real> y_exact = {
    real(1),
    real(-2),
    real(3),
    real(-1)
  };

  std::vector<real> g(n, real(0));
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      g[i] += R[index2D(i, j, n)] * y_exact[j];
    }
  }

  const auto y = detail::solveUpperTriangular(R, g);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(y[i], y_exact[i], 1e-12);
  }
}

TEST(GMRES, solveUpperTriangularComplex) {
  const int n = 4;

  std::vector<complex> R(n * n, complex(0));

  R[index2D(0, 0, n)] = complex( 2,  1);
  R[index2D(0, 1, n)] = complex(-1,  2);
  R[index2D(0, 2, n)] = complex( 3, -1);
  R[index2D(0, 3, n)] = complex( 1,  0);

  R[index2D(1, 1, n)] = complex( 4, -1);
  R[index2D(1, 2, n)] = complex(-2,  3);
  R[index2D(1, 3, n)] = complex( 2,  1);

  R[index2D(2, 2, n)] = complex( 5,  2);
  R[index2D(2, 3, n)] = complex(-3,  1);

  R[index2D(3, 3, n)] = complex(-2, -4);

  const std::vector<complex> y_exact = {
    complex( 1,  2),
    complex(-2,  1),
    complex( 3, -1),
    complex(-1,  2)
  };

  std::vector<complex> g(n, complex(0));

  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      g[i] += R[index2D(i, j, n)] * y_exact[j];
    }
  }

  const auto y = detail::solveUpperTriangular(R, g);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(y[i].real(), y_exact[i].real(), 1e-12);
    EXPECT_NEAR(y[i].imag(), y_exact[i].imag(), 1e-12);
  }
}

TEST(GMRES, runArnoldiStepOrthonormalBase) {
  const int N = 6;
  const int memory = 4;
  cudaStream_t st = 0;

  detail::Workspace<real> work(N, memory);

  auto& krylovBase = work.krylovBase;
  auto& H          = work.hessenberg;

  std::fill(H.begin(), H.end(), real(0));

  std::vector<real> v0_h = {
    real(1),
    real(2),
    real(-1),
    real(0.5),
    real(3),
    real(-2)
  };

  real nrm = real(0);
  for (const real x : v0_h) {
    nrm += x * x;
  }

  nrm = std::sqrt(nrm);

  for (real& x : v0_h) {
    x /= nrm;
  }

  thrust::device_vector<real> v0(v0_h);

  writeColumn(v0, krylovBase, N, 0, st);

  struct Operator {
    int N;

    void operator()(const thrust::device_vector<real>& v,
                    thrust::device_vector<real>& Av,
                    cudaStream_t st) const {
      (void) st;

      thrust::host_vector<real> hv = v;
      thrust::host_vector<real> hAv(N, real(0));

      for (int i = 0; i < N; ++i) {
        hAv[i] = real(2) * hv[i];

        if (i > 0) {
          hAv[i] -= hv[i - 1];
        }

        if (i < N - 1) {
          hAv[i] -= hv[i + 1];
        }
      }

      Av = hAv;
    }
  };

  Operator op{N};

  thrust::device_vector<real> va(N);
  thrust::device_vector<real> vb(N);

  for (int j = 0; j < memory; ++j) {
    const bool ok = detail::runArnoldiStep(op, work, j, st);
    
    ASSERT_TRUE(ok);
    
    for (int a = 0; a <= j + 1; ++a) {
      getColumn(krylovBase, va, a, st);
      
      EXPECT_NEAR(norm(va, st), real(1), 1e-12);
      
      for (int b = a + 1; b <= j + 1; ++b) {
        getColumn(krylovBase, vb, b, st);
        
        const real ip = dotc(va, vb, st);
        EXPECT_NEAR(ip, real(0), 1e-12);
      }
    }
  }
}

TEST(GMRES, runArnoldiStepBreakdown) {
  const int N = 4;
  const int memory = 3;
  cudaStream_t st = 0;

  detail::Workspace<real> work(N, memory);

  auto& krylovBase = work.krylovBase;
  auto& H          = work.hessenberg;

  std::fill(H.begin(), H.end(), real(0));

  thrust::device_vector<real> v0 = {
    real(1),
    real(0),
    real(0),
    real(0)
  };

  writeColumn(v0, krylovBase, N, 0, st);

  struct IdentityOperator {
    int N;

    void operator()(const thrust::device_vector<real>& v,
                    thrust::device_vector<real>& out,
                    cudaStream_t st) const {
      (void) st;

      thrust::copy(v.begin(), v.end(), out.begin());
    }
  };

  IdentityOperator identity{N};
  
  const bool ok = detail::runArnoldiStep(identity, work, 0, st);
  
  EXPECT_FALSE(ok);
  EXPECT_NEAR(H[index2D(1, 0, memory)], real(0), 1e-12);
}

TEST(GMRES, runArnoldiStepOrthonormalBaseComplex) {
  const int N = 6;
  const int memory = 4;
  cudaStream_t st = 0;

  detail::Workspace<complex> work(N, memory);

  auto& krylovBase = work.krylovBase;
  auto& H          = work.hessenberg;

  std::fill(H.begin(), H.end(), complex(0.0, 0.0));

  std::vector<complex> v0_h = {
    complex( 1.0,  0.5),
    complex( 2.0, -1.0),
    complex(-1.0,  0.25),
    complex( 0.5,  1.5),
    complex( 3.0, -0.75),
    complex(-2.0,  0.5)
  };

  real nrm = real(0);
  for (const complex& x : v0_h) {
    nrm += thrust::norm(x);
  }

  nrm = std::sqrt(nrm);

  for (complex& x : v0_h) {
    x /= nrm;
  }

  thrust::device_vector<complex> v0(v0_h);
  writeColumn(v0, krylovBase, N, 0, st);

  struct Operator {
    int N;

    void operator()(const thrust::device_vector<complex>& v,
                    thrust::device_vector<complex>& Av,
                    cudaStream_t st) const {
      (void) st;

      thrust::host_vector<complex> hv = v;
      thrust::host_vector<complex> hAv(N, complex(0.0, 0.0));

      for (int i = 0; i < N; ++i) {
        hAv[i] = complex(2.0, 0.0) * hv[i];

        if (i > 0) {
          hAv[i] -= complex(1.0, 0.5) * hv[i - 1];
        }

        if (i < N - 1) {
          hAv[i] -= complex(1.0, -0.25) * hv[i + 1];
        }
      }

      Av = hAv;
    }
  };

  Operator op{N};

  thrust::device_vector<complex> va(N);
  thrust::device_vector<complex> vb(N);

  for (int j = 0; j < memory; ++j) {
     const bool ok = detail::runArnoldiStep(op, work, j, st);

    ASSERT_TRUE(ok);

    for (int a = 0; a <= j + 1; ++a) {
      getColumn(krylovBase, va, a, st);

      EXPECT_NEAR(norm(va, st), real(1), 1e-12);

      for (int b = a + 1; b <= j + 1; ++b) {
        getColumn(krylovBase, vb, b, st);

        const complex dab = dotc(va, vb, st);

        EXPECT_NEAR(dab.real(), real(0), 1e-12);
        EXPECT_NEAR(dab.imag(), real(0), 1e-12);
      }
    }
  }
}

TEST(GMRES, runArnoldiStepBreakdownComplex) {
  const int N = 4;
  const int memory = 3;
  cudaStream_t st = 0;

  detail::Workspace<complex> work(N, memory);

  auto& krylovBase = work.krylovBase;
  auto& H          = work.hessenberg;

  std::fill(H.begin(), H.end(), complex(0.0, 0.0));

  thrust::device_vector<complex> v0 = {
    complex(1.0, 0.0),
    complex(0.0, 0.0),
    complex(0.0, 0.0),
    complex(0.0, 0.0)
  };

  writeColumn(v0, krylovBase, N, 0, st);

  struct IdentityOperator {
    void operator()(const thrust::device_vector<complex>& v,
                    thrust::device_vector<complex>& out,
                    cudaStream_t st) const {
      (void) st;

      thrust::copy(v.begin(), v.end(), out.begin());
    }
  };

  IdentityOperator identity;

   const bool ok = detail::runArnoldiStep(identity, work, 0, st);
  
   EXPECT_FALSE(ok);

   const complex h10 = H[index2D(1, 0, memory)];
   
   EXPECT_NEAR(h10.real(), real(0), 1e-12);
   EXPECT_NEAR(h10.imag(), real(0), 1e-12);
}

TEST(GMRES, solveHappyBreakdownAfterPreviousRotation) {
  struct DenseOperator {
    std::vector<real> A;
    int n;

    void operator()(const thrust::device_vector<real>& x,
                    thrust::device_vector<real>& out,
                    cudaStream_t st) const {
      (void) st;

      std::vector<real> hx(n, real(0));
      thrust::copy(x.begin(), x.end(), hx.begin());

      std::vector<real> hy(n, real(0));

      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          hy[i] += A[i * n + j] * hx[j];
        }
      }

      thrust::copy(hy.begin(), hy.end(), out.begin());
    }
  };

  const int n = 2;

  std::vector<real> A = {
    real(3), real(1),
    real(2), real(4)
  };

  std::vector<real> xtrue_h = {
    real(2),
    real(-1)
  };

  std::vector<real> b_h(n, real(0));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      b_h[i] += A[i * n + j] * xtrue_h[j];
    }
  }

  thrust::device_vector<real> b(b_h);
  thrust::device_vector<real> x0(n, real(0));
  
  Parameters p;
  p.memory        = 4;
  p.maxIterations = 1;
  p.tolerance     = real(1e-14);
  
  const auto res = solve(DenseOperator{A, n}, b, x0, p, 0);
  
  EXPECT_TRUE(res.info.converged);
  EXPECT_EQ(res.info.requiredIterations, 2);
  
  std::vector<real> x_h(n, real(0));
  thrust::copy(res.x.begin(), res.x.end(), x_h.begin());
  
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(x_h[i], xtrue_h[i], 1e-10);
  }
  
  EXPECT_LT(res.info.relativeError, 1e-12);
}


TEST(GMRES, solveRealSystem) {
  struct DenseOperator {
    std::vector<real> A;
    int n;
    
    void operator()(const thrust::device_vector<real>& x,
                    thrust::device_vector<real>& out,
                    cudaStream_t st) const {
      std::vector<real> hx(n, real(0));
      thrust::copy(x.begin(), x.end(), hx.begin());
      
      std::vector<real> hy(n, real(0));
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          hy[i] += A[i * n + j] * hx[j];
        }
      }

      thrust::copy(hy.begin(), hy.end(), out.begin());
    }
  };
  
  const int n = 3;
  
  std::vector<real> A = {
    real(4.0), real(1.0), real(0.0),
    real(2.0), real(3.0), real(1.0),
    real(0.0), real(1.0), real(2.0)
  };
  
  std::vector<real> xtrue_h = {
    real(1.0),
    real(2.0),
    real(3.0)
  };
  
  // b = A * x_true
  std::vector<real> b_h(n, real(0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      b_h[i] += A[i * n + j] * xtrue_h[j];
    }
  }

  thrust::device_vector<real> b(b_h);
  thrust::device_vector<real> x0(n, real(0));
  
  DenseOperator op{A, n};
  
  Parameters p;
  p.memory        = 5;
  p.maxIterations = 20;
  p.tolerance     = real(1e-10);
  
  cudaStream_t st = 0;
  
  auto res = solve(op, b, x0, p, st);
  
  EXPECT_TRUE(res.info.converged);
  
  std::vector<real> x_h(n, real(0));
  thrust::copy(res.x.begin(), res.x.end(), x_h.begin());
  
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(x_h[i], xtrue_h[i], 1e-8);
  }
  
  std::vector<real> Ax_h(n, real(0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Ax_h[i] += A[i * n + j] * x_h[j];
    }
  }

  real resnorm2 = real(0);
  for (int i = 0; i < n; ++i) {
    const real ri = b_h[i] - Ax_h[i];
    resnorm2 += ri * ri;
  }

  const real resnorm = std::sqrt(resnorm2);

  EXPECT_NEAR(resnorm, real(0), 1e-8);
  EXPECT_LT(res.info.relativeError, 1e-8);
}

TEST(GMRES, solveComplexSystem) {
  struct DenseOperator {
    std::vector<complex> A;
    int n;
    
    void operator()(const thrust::device_vector<complex>& x,
                    thrust::device_vector<complex>& out,
                    cudaStream_t st) const {
      std::vector<complex> hx(n, complex(0));
      thrust::copy(x.begin(), x.end(), hx.begin());
      
      std::vector<complex> hy(n, complex(0));
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          hy[i] += A[i * n + j] * hx[j];
        }
      }
      thrust::copy(hy.begin(), hy.end(), out.begin());
    }
  };
  
  const int n = 3;
  
  std::vector<complex> A = {
    complex(4.0,  1.0), complex(1.0, -0.5), complex(0.0,  0.0),
    complex(2.0, -1.0), complex(3.0,  0.5), complex(1.0,  2.0),
    complex(0.0,  0.0), complex(1.0, -1.5), complex(2.0, -0.5)
  };
  
  std::vector<complex> xtrue_h = {
    complex(1.0,  0.5),
    complex(2.0, -1.0),
    complex(3.0,  2.0)
  };
  
  // b = A * x_true
  std::vector<complex> b_h(n, complex(0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      b_h[i] += A[i * n + j] * xtrue_h[j];
    }
  }

  thrust::device_vector<complex> b(b_h);
  thrust::device_vector<complex> x0(n, complex(0));
  
  DenseOperator op{A, n};
  
  Parameters p;
  p.memory        = 5;
  p.maxIterations = 20;
  p.tolerance     = real(1e-10);
  
  cudaStream_t st = 0;
  
  auto res = solve(op, b, x0, p, st);
  
  EXPECT_TRUE(res.info.converged);
  
  std::vector<complex> x_h(n, complex(0));
  thrust::copy(res.x.begin(), res.x.end(), x_h.begin());
  
  for (int i = 0; i < n; ++i) {   
    EXPECT_NEAR(x_h[i].real(), xtrue_h[i].real(), 1e-8);
    EXPECT_NEAR(x_h[i].imag(), xtrue_h[i].imag(), 1e-8);
  }
  
  std::vector<complex> Ax_h(n, complex(0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Ax_h[i] += A[i * n + j] * x_h[j];
    }
  }

  real resnorm2 = real(0);
  for (int i = 0; i < n; ++i) {
    const complex ri = b_h[i] - Ax_h[i];
    resnorm2 += thrust::norm(ri);
  }

  const real resnorm = std::sqrt(resnorm2);

  EXPECT_NEAR(resnorm, 0.0, 1e-8);
  EXPECT_LT(res.info.relativeError, 1e-8);
}

