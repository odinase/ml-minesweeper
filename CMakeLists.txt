cmake_minimum_required(VERSION 3.4...3.18)
project(pybind_testing)

# find_package( Eigen3 REQUIRED eigen3 )

add_subdirectory(pybind11)
include_directories( ${PROJECT_SOURCE_DIR}/eigen )
pybind11_add_module(pybind_testing module.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(pybind_testing PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
