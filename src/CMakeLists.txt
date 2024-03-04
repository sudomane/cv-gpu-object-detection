cmake_minimum_required(VERSION 3.18.4)
project(ObjDetect)

set(SOURCE_FILES
    main.cpp
    pipeline.cpp
    grayscale.cpp
    difference.cpp
    gaussian.cpp
    morphology.cpp
    binary.cpp
    components.cpp
    bbox.cpp
)

add_executable(main ${SOURCE_FILES})

# Add debugging flags for Debug configuration
target_compile_options(main PRIVATE
    $<$<CONFIG:Debug>:-O0 -g -Wall -Wextra>
    $<$<CONFIG:Release>:-O3>
)



find_package(OpenCV REQUIRED)

target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ext/json/include)

target_link_libraries(main PRIVATE ${OpenCV_LIBS})

enable_language(CUDA)

set_property(TARGET main PROPERTY CXX_STANDARD 20)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 75)