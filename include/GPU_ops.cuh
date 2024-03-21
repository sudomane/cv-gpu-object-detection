#pragma once

#include <tuple>
#include <vector>

typedef std::pair<int,int> t_point;

namespace GPU
{
    __global__ void grayscale  (unsigned char* d_dst, const unsigned char* d_src, int width, int height);
    __global__ void difference (unsigned char* d_dst, const unsigned char* d_src, int width, int height);
    __global__ void gaussian   (unsigned char* d_dst, int width, int height, int kernel_size, int sigma);

}; // namespace cpu