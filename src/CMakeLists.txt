cmake_minimum_required(VERSION 3.18.4)
project(ObjDetect)

set(SOURCE_FILES
    main.cpp

    pipeline.cu
    pipeline.cpp

    cpu/grayscale.cpp
    cpu/difference.cpp
    cpu/gaussian.cpp
    cpu/morphology.cpp
    cpu/binary.cpp
    cpu/components.cpp
    cpu/bbox.cpp

    gpu/grayscale.cu
    gpu/difference.cu
    gpu/gaussian.cu
    gpu/morphology.cu
    gpu/binary.cu
    gpu/components.cu
    gpu/bbox.cu
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipeline ${SOURCE_FILES})

# Add debugging flags for Debug configuration
target_compile_options(pipeline PRIVATE
    $<$<CONFIG:Debug>:-O0 -g -Wall -Wextra>
    $<$<CONFIG:Release>:-O3>
)

find_package(OpenCV REQUIRED)

target_include_directories(pipeline PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ext/json/include)

target_link_libraries(pipeline PRIVATE ${OpenCV_LIBS})

enable_language(CUDA)

set_property(TARGET pipeline PROPERTY CXX_STANDARD 20)
set_property(TARGET pipeline PROPERTY CUDA_ARCHITECTURES 75)