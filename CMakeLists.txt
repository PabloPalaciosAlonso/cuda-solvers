cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda-solvers LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build C++ tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(DOUBLE_PRECISION "Use double precision" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(cuda_solvers INTERFACE)

target_include_directories(cuda_solvers
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(cuda_solvers
  INTERFACE
    cxx_std_17
)

if(DOUBLE_PRECISION)
  target_compile_definitions(cuda_solvers INTERFACE DOUBLE_PRECISION)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
