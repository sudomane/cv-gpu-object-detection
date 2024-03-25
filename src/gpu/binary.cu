#include <GPU_ops.cuh>

__global__ void GPU::binary(unsigned char* d_data, int bin_thresh, int width, int height)
{
    int dim = blockDim.x * blockIdx.x + threadIdx.x;

    int x = dim % width;
    int y = dim / width;

    if (x >= width || y >= height)
        return;

    d_data[dim]= d_data[dim] > bin_thresh ? 255 : 0;
}