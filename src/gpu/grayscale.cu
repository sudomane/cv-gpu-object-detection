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


void GPU::HostWrapper::grayscale(unsigned char* d_dst, const unsigned char* d_src)
{
    GPU::grayscale<<<num_blocks, block_size>>>(d_dst, d_src, this->width, this->height);

    cudaDeviceSynchronize();

    cudaError_t error = cudaPeekAtLastError();

    if (error != cudaSuccess)
        errx(1, "[grayscale] CUDA Error: %s\n", cudaGetErrorString(error));
}