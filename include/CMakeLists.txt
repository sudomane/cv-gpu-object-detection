cmake_minimum_required(VERSION 3.18.4)
project(ObjDetect)

# Adding the include directory to list of directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(HEADER_FILES
    pipeline.hpp,
    CPU_ops.hpp,
    GPU_ops.cuh,
)

