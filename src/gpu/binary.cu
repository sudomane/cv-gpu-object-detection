#include <GPU_ops.cuh>

__global__ void GPU::binary(unsigned char* d_dst, int bin_thresh, int width, int height)
{
    int dim = blockDim.x * blockIdx.x + threadIdx.x;

    int x = dim % width;
    int y = dim / width;

    if (x >= width || y >= height)
        return;

    d_dst[dim]= d_dst[dim] > bin_thresh ? 255 : 0;
}