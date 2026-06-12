include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.16.0
)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

file(GLOB_RECURSE TEST_SOURCES
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})

    target_link_libraries(${test_name}
      PRIVATE
        cuda_solvers
        GTest::gtest_main
    )

    gtest_discover_tests(${test_name})
endforeach()
