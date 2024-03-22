#pragma once

#include <tuple>
#include <vector>

typedef std::pair<int,int> t_point;

__device__ inline void _getDeviceBoundaries(int i, int kernel_size, int size, int offset, int& start, int& end)
{
    int _start = 0;
    int _end   = size;

    if (i > offset)
        _start = i - offset;

    if (i + offset < size)
        _end = i + offset + 1;

    start = _start;
    end   = _end;
}


namespace GPU
{
    __global__ void grayscale  (unsigned char* d_dst, const unsigned char* d_src, int width, int height);
    __global__ void difference (unsigned char* d_dst, const unsigned char* d_src, int width, int height);
    __global__ void gaussian   (unsigned char* d_dst, const unsigned char* d_src, const float* d_kernel, int width, int height, int kernel_size, int sigma, int offset);
    __global__ void morphology (unsigned char* d_dst, unsigned char* d_src, int width, int height, int opening_size, int closing_size, int offset);
    __global__ void binary     (unsigned char* d_dst, int bin_thresh, int width, int height);
}; // namespace cpu