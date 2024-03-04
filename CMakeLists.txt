cmake_minimum_required(VERSION 3.18.4)

project(ObjDetect LANGUAGES CUDA CXX)
add_subdirectory(ext)
add_subdirectory(include)
add_subdirectory(src)