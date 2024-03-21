#include <GPU_ops.cuh>

__global__ void GPU::grayscale(unsigned char* d_dst, const unsigned char* d_src, int width, int height)
{
    int dim = blockDim.x * blockIdx.x + threadIdx.x;

    int x = dim % width;
    int y = dim / width;

    if (x >= width || y >= height)
        return;

    int r = d_src[dim * 3];
    int g = d_src[dim * 3 + 1];
    int b = d_src[dim * 3 + 2];

    d_dst[dim] = static_cast<unsigned char>((r + g + b) / 3);
}