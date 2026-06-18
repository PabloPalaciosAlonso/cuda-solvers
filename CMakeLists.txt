cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda-solvers LANGUAGES CXX CUDA)

option(CUDA_SOLVERS_BUILD_TESTS "Build C++ tests" ${PROJECT_IS_TOP_LEVEL})
option(DOUBLE_PRECISION "Use double precision" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_solvers INTERFACE)

target_include_directories(cuda_solvers
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


target_link_libraries(cuda_solvers
  INTERFACE
  CUDA::cusolver
  CUDA::cublas
)
target_compile_features(cuda_solvers
  INTERFACE
    cxx_std_17
)

if(DOUBLE_PRECISION)
  target_compile_definitions(cuda_solvers INTERFACE DOUBLE_PRECISION)
endif()

if(CUDA_SOLVERS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
