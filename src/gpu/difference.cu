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

void GPU::HostWrapper::difference(unsigned char* d_dst, const unsigned char* d_src)
{
    GPU::difference<<<this->num_blocks, this->block_size>>>(d_dst, d_src, this->width, this->height);

    cudaDeviceSynchronize();

    cudaError_t error = cudaPeekAtLastError();

    if (error != cudaSuccess)
        errx(1, "[difference] CUDA Error: %s\n", cudaGetErrorString(error));
}