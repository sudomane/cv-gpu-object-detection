#pragma once

#include <tuple>
#include <vector>

typedef std::pair<int,int> t_point;

namespace GPU
{
    __global__ void grayscale  (unsigned char* d_dst, const unsigned char* d_src, int width, int height);
    __global__ void difference (unsigned char* d_dst, const unsigned char* d_src, int width, int height);
    __global__ void gaussian   (unsigned char* d_dst, const unsigned char* d_src, const float* d_kernel, int width, int height, int kernel_size, int sigma, int offset);
    __global__ void morphology (unsigned char* d_dst, const unsigned char* d_src, int width, int height, int opening_size, int closing_size)

}; // namespace cpu