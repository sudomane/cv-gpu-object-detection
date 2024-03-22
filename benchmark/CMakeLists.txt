cmake_minimum_required(VERSION 3.18.4)

project(ObjDetect)

find_package(OpenCV REQUIRED)
find_package(benchmark REQUIRED)

add_executable(bench
    benchmark.cpp

    ../src/cpu/grayscale.cpp
    ../src/cpu/difference.cpp
    ../src/cpu/gaussian.cpp
    ../src/cpu/morphology.cpp
    ../src/cpu/binary.cpp
    ../src/cpu/components.cpp
    ../src/cpu/bbox.cpp
)

target_include_directories(bench PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(bench PRIVATE ${OpenCV_LIBS} benchmark::benchmark)

set_property(TARGET bench PROPERTY CXX_STANDARD 20)

## GPU BENCHMARK

enable_language(CUDA)

add_executable(bench_gpu
    benchmark.cu

    ../src/gpu/grayscale.cu
    ../src/gpu/difference.cu
    ../src/gpu/gaussian.cu
    ../src/gpu/morphology.cu
    ../src/gpu/binary.cu
    ../src/gpu/components.cu
    ../src/gpu/bbox.cu
)

target_include_directories(bench_gpu PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(bench_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(bench_gpu PRIVATE ${OpenCV_LIBS} benchmark::benchmark)

set_property(TARGET bench_gpu PROPERTY CXX_STANDARD 20)
set_property(TARGET bench_gpu PROPERTY CUDA_ARCHITECTURES 75)