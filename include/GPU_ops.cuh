#pragma once

#include <err.h>
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

template<typename T>
inline T* _cudaMalloc(size_t n)
{
    T* d_out;

    int rc = cudaMalloc(&d_out, sizeof(T) * n);
    if (rc)
      errx(1, "Failed to allocate d_out.");

    return d_out;
}

template<typename T>
inline T* _toDevice(const T* h_src, int width, int height, int n_channels = 1)
{
    T* d_dst = _cudaMalloc<T>(width * height * n_channels);

    int rc = cudaMemcpy(d_dst, h_src, sizeof(T) * width * height * n_channels, cudaMemcpyHostToDevice);
    if (rc)
        errx(1, "Failed to copy host memory to device buffer.");

    return d_dst;
}

inline float* _generateDeviceKernel(int kernel_size, float sigma)
{
    float  sum = 0;
    float* h_kernel = new float[kernel_size * kernel_size];

    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            float x = i - kernel_size / 2;
            float y = j - kernel_size / 2;

            float val = std::exp2f(-(x*x + y*y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            h_kernel[j * kernel_size + i] = val;

            sum += val;
        }
    }

    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            h_kernel[j * kernel_size + i] /= sum;
        }
    }

    float* d_kernel = _toDevice<float>(h_kernel, kernel_size, kernel_size);

    delete[] h_kernel;

    return d_kernel;
}

namespace GPU
{
    __global__ void grayscale  (unsigned char* d_dst, const unsigned char* d_src, int width, int height);
    __global__ void difference (unsigned char* d_dst, const unsigned char* d_src, int width, int height);
    __global__ void gaussian   (unsigned char* d_dst, const unsigned char* d_src, const float* d_kernel, int width, int height, int kernel_size, int offset);
    __global__ void morphology (unsigned char* d_dst, unsigned char* d_src, int width, int height, int opening_size, int closing_size, int offset);
    __global__ void binary     (unsigned char* d_dst, int bin_thresh, int width, int height);
    __global__ void components (unsigned char* d_dst, int width, int height);
    __global__ void getBbox(unsigned char* d_src, int* coords, int width, int height, int label);
}; // namespace gpu