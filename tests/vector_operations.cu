#include <gtest/gtest.h>
#include "cuda_solvers/types.h"
#include "cuda_solvers/detail/vector_operations.cuh"

using namespace cuda_solvers;

TEST(VECTOR_OPERATIONS_REAL, substract)
{
  thrust::device_vector<real> v1 = {1.0, 2.0, 3.0};
  thrust::device_vector<real> v2 = {-2.1, 1.2, -3.5};
  thrust::device_vector<real> v3(v2.size());

  real tol = 1e-12;
  substract(v1, v2, v3);

  for(int i = 0; i<v1.size(); i++)
    EXPECT_NEAR(v1[i]-v2[i], v3[i], tol);
}

TEST(VECTOR_OPERATIONS_COMPLEX, substract){

  complex ii{0, 1};
  complex one{1, 0};
  
  thrust::device_vector<complex> v1 = {
    3.1 * one + 2.1 * ii,
    2.0 * one - 1.3 * ii,
    3.0 * one + 4.2 * ii
  };

  thrust::device_vector<complex> v2 = {
    -2.1 * one + 0.5 * ii,
     1.2 * one - 2.0 * ii,
    -3.5 * one + 1.1 * ii
  };

  thrust::device_vector<complex> v3(v2.size());

  real tol = 1e-12;

  substract(v1, v2, v3);

  for(int i = 0; i < v1.size(); i++) {
    complex expected = complex(v1[i]) - complex(v2[i]);

    EXPECT_NEAR(expected.real(), complex(v3[i]).real(), tol);
    EXPECT_NEAR(expected.imag(), complex(v3[i]).imag(), tol);
  }
}

TEST(VECTOR_OPERATIONS_REAL, add){
thrust::device_vector<real> v1 = {1.0, 2.0, 3.0};
thrust::device_vector<real> v2 = {-2.1, 1.2, -3.5};
thrust::device_vector<real> v3(v2.size());

real tol = 1e-12;
add(v1, v2, v3);

for(int i = 0; i<v1.size(); i++)
  EXPECT_NEAR(v1[i]+v2[i], v3[i], tol);
}

TEST(VECTOR_OPERATIONS_COMPLEX, add){
  complex ii{0, 1};
  complex one{1, 0};
  thrust::device_vector<complex> v1 = {
    3.1 * one + 2.1 * ii,
    2.0 * one - 1.3 * ii,
    3.0 * one + 4.2 * ii
  };
  thrust::device_vector<complex> v2 = {
    -2.1 * one + 0.5 * ii,
     1.2 * one - 2.0 * ii,
    -3.5 * one + 1.1 * ii
  };
  thrust::device_vector<complex> v3(v2.size());
  real tol = 1e-12;
  add(v1, v2, v3);
  for(int i = 0; i < v1.size(); i++) {
    complex expected = complex(v1[i]) + complex(v2[i]);
    EXPECT_NEAR(expected.real(), complex(v3[i]).real(), tol);
    EXPECT_NEAR(expected.imag(), complex(v3[i]).imag(), tol);
  }
}

TEST(VECTOR_OPERATIONS_REAL, multiply_vector_by_vector)
{
  thrust::device_vector<real> v1 = {1.0, 2.0, 3.0};
  thrust::device_vector<real> v2 = {-2.1, 1.2, -3.5};
  thrust::device_vector<real> v3(v2.size());

  real tol = 1e-12;
  multiply(v1, v2, v3);
  for (int i = 0; i < v1.size(); i++)
    EXPECT_NEAR(v1[i] * v2[i], v3[i], tol);
}

TEST(VECTOR_OPERATIONS_COMPLEX, multiply_vector_by_vector)
{
  complex ii{0, 1};
  complex one{1, 0};
  thrust::device_vector<complex> v1 = {
      3.1 * one + 2.1 * ii,
      2.0 * one - 1.3 * ii,
      3.0 * one + 4.2 * ii};
  thrust::device_vector<complex> v2 = {
      -2.1 * one + 0.5 * ii,
      1.2 * one - 2.0 * ii,
      -3.5 * one + 1.1 * ii};
  thrust::device_vector<complex> v3(v2.size());
  real tol = 1e-12;
  multiply(v1, v2, v3);
  for (int i = 0; i < v1.size(); i++)
  {
    complex expected = complex(v1[i]) * complex(v2[i]);
    EXPECT_NEAR(expected.real(), complex(v3[i]).real(), tol);
    EXPECT_NEAR(expected.imag(), complex(v3[i]).imag(), tol);
  }
}

TEST(VECTOR_OPERATIONS_REAL, multiply_vector_by_scalar){
  thrust::device_vector<real> v1 = {1.0, 2.0, 3.0};
  thrust::device_vector<real> v2(v1.size());
  real alpha = 2.5; 

  real tol = 1e-12;
  multiply(v1, alpha, v2); 
  for(int i = 0; i<v1.size(); i++)
    EXPECT_NEAR(alpha * v1[i], v2[i], tol);
}

TEST(VECTOR_OPERATIONS_COMPLEX, multiply_vector_by_scalar){
  complex ii{0, 1};
  complex one{1, 0};
  thrust::device_vector<complex> v1 = {
    3.1 * one + 2.1 * ii,
    2.0 * one - 1.3 * ii,
    3.0 * one + 4.2 * ii
  };
  thrust::device_vector<complex> v2(v1.size());
  complex alpha = complex(2.5, -1.5); 

  real tol = 1e-12;
  multiply(v1, alpha, v2);
  for(int i = 0; i < v1.size(); i++) {
    complex expected = alpha * complex(v1[i]);
    EXPECT_NEAR(expected.real(), complex(v2[i]).real(), tol);
    EXPECT_NEAR(expected.imag(), complex(v2[i]).imag(), tol);
  }
}

TEST(VECTOR_OPERATIONS_REAL, dividie_vector_by_vector){
  thrust::device_vector<real> v1 = {1.0, 2.0, 3.0};
  thrust::device_vector<real> v2 = {-2.1, 1.2, -3.5};
  thrust::device_vector<real> v3(v2.size());

  real tol = 1e-12;
  divide(v1, v2, v3);
  for(int i = 0; i<v1.size(); i++)
    EXPECT_NEAR(v1[i] / v2[i], v3[i], tol);
}

TEST(VECTOR_OPERATIONS_COMPLEX, dividie_vector_by_vector){
  complex ii{0, 1};
  complex one{1, 0};
  thrust::device_vector<complex> v1 = {
    3.1 * one + 2.1 * ii,
    2.0 * one - 1.3 * ii,
    3.0 * one + 4.2 * ii
  };
  thrust::device_vector<complex> v2 = {
    -2.1 * one + 0.5 * ii,
     1.2 * one - 2.0 * ii,
    -3.5 * one + 1.1 * ii
  };
  thrust::device_vector<complex> v3(v2.size());
  real tol = 1e-12;
  divide(v1, v2, v3);
  for(int i = 0; i < v1.size(); i++) {
    complex expected = complex(v1[i]) / complex(v2[i]);
    EXPECT_NEAR(expected.real(), complex(v3[i]).real(), tol);
    EXPECT_NEAR(expected.imag(), complex(v3[i]).imag(), tol);
  }
}

TEST(VECTOR_OPERATIONS_REAL, divide_vector_by_scalar){
  thrust::device_vector<real> v1 = {1.0, 2.0, 3.0};
  thrust::device_vector<real> v2(v1.size());
  real alpha = 2.5; 

  real tol = 1e-12;
  divide(v1, alpha, v2); 
  for(int i = 0; i<v1.size(); i++)
    EXPECT_NEAR(v1[i] / alpha, v2[i], tol);
} 

TEST(VECTOR_OPERATIONS_COMPLEX, divide_vector_by_scalar){
  complex ii{0, 1};
  complex one{1, 0};
  thrust::device_vector<complex> v1 = {
    3.1 * one + 2.1 * ii,
    2.0 * one - 1.3 * ii,
    3.0 * one + 4.2 * ii
  };
  thrust::device_vector<complex> v2(v1.size());
  complex alpha = complex(2.5, -1.5); 

  real tol = 1e-12;
  divide(v1, alpha, v2);
  for(int i = 0; i < v1.size(); i++) {
    complex expected = complex(v1[i]) / alpha;
    EXPECT_NEAR(expected.real(), complex(v2[i]).real(), tol);
    EXPECT_NEAR(expected.imag(), complex(v2[i]).imag(), tol);
  }
}

TEST(VECTOR_OPERATIONS_REAL, axpy){
  thrust::device_vector<real> x = {1.0, 2.0, 3.0};
  thrust::device_vector<real> y = {-2.1, 1.2, -3.5};
  thrust::device_vector<real> out(y.size());
  real alpha = 2.5; 

  real tol = 1e-12;
  axpy(x, y, out, alpha); 
  for(int i = 0; i<x.size(); i++)
    EXPECT_NEAR(alpha * x[i] + y[i], out[i], tol);
}

TEST(VECTOR_OPERATIONS_COMPLEX, axpy){
  complex ii{0, 1};
  complex one{1, 0};
  thrust::device_vector<complex> x = {
    3.1 * one + 2.1 * ii,
    2.0 * one - 1.3 * ii,
    3.0 * one + 4.2 * ii
  };
  thrust::device_vector<complex> y = {
    -2.1 * one + 0.5 * ii,
     1.2 * one - 2.0 * ii,
    -3.5 * one + 1.1 * ii
  };
  thrust::device_vector<complex> out(y.size());
  complex alpha = complex(2.5, -1.5); 

  real tol = 1e-12;
  axpy(x, y, out, alpha);
  for(int i = 0; i < x.size(); i++) {
    complex expected = alpha * complex(x[i]) + complex(y[i]);
    EXPECT_NEAR(expected.real(), complex(out[i]).real(), tol);
    EXPECT_NEAR(expected.imag(), complex(out[i]).imag(), tol);
  }
}

TEST(VECTOR_OPERATIONS_REAL, dotc){
  thrust::device_vector<real> v1 = {1.0, 2.0, 3.0};
  thrust::device_vector<real> v2 = {-2.1, 1.2, -3.5};

  real tol = 1e-12;
  real result = dotc(v1, v2);
  real expected = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
  EXPECT_NEAR(expected, result, tol);
}

TEST(VECTOR_OPERATIONS_COMPLEX, dotc){
  complex ii{0, 1};
  complex one{1, 0};
  thrust::device_vector<complex> v1 = {
    3.1 * one + 2.1 * ii,
    2.0 * one - 1.3 * ii,
    3.0 * one + 4.2 * ii
  };
  thrust::device_vector<complex> v2 = {
    -2.1 * one + 0.5 * ii,
     1.2 * one - 2.0 * ii,
    -3.5 * one + 1.1 * ii
  };

  real tol = 1e-12;
  complex result = dotc(v1, v2);
  
  complex expected = complex(0,0);
  for(int i = 0; i < v1.size(); i++)
    expected += thrust::conj(complex(v1[i])) * complex(v2[i]);

  EXPECT_NEAR(expected.real(), result.real(), tol);
  EXPECT_NEAR(expected.imag(), result.imag(), tol);
}

TEST(VECTOR_OPERATIONS_REAL, norm){
  thrust::device_vector<real> v = {3.0, 4.0};

  real tol = 1e-12;
  real result = norm(v);
  real expected = 5.0; 
  EXPECT_NEAR(expected, result, tol);
}

TEST(VECTOR_OPERATIONS_COMPLEX, norm){
  complex ii{0, 1};
  complex one{1, 0};
  thrust::device_vector<complex> v = {
    3.0 * one + 4.0 * ii
  };

  real tol = 1e-12;
  real result = norm(v);
  real expected = 5.0; 
  EXPECT_NEAR(expected, result, tol);
} 
