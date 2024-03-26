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

void GPU::HostWrapper::binary(unsigned char* d_dst)
{
    GPU::binary<<<this->num_blocks, this->block_size>>>(d_dst, this->bin_thresh, this->width, this->height);

    cudaDeviceSynchronize();

    cudaError_t error = cudaPeekAtLastError();

    if (error != cudaSuccess)
        errx(1, "[binary] CUDA Error: %s\n", cudaGetErrorString(error));
}