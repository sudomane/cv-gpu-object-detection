#include <GPU_ops.cuh>

#include <cmath>

__global__ void GPU::difference(unsigned char* d_dst, const unsigned char* d_src, int width, int height)
{
    int dim = blockDim.x * blockIdx.x + threadIdx.x;

    int x = dim % width;
    int y = dim / width;

    if (x >= width || y >= height)
        return;

    d_dst[dim] = abs(d_src[dim] - d_dst[dim]);
}