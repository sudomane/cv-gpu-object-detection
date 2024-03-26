#include <GPU_ops.cuh>

__global__ void GPU::getBbox(int* d_labels, int* coords, int width, int height, int label)
{
    // fixme
}

void GPU::HostWrapper::getBbox(int* d_labels, int* coords, int label)
{
    // TODO: Updated
    GPU::getBbox<<<this->num_blocks, this->block_size>>>(d_labels, nullptr, this->width, this->height, 0);

    cudaDeviceSynchronize();

    cudaError_t error = cudaPeekAtLastError();

    if (error != cudaSuccess)
        errx(1, "[getBbox] CUDA Error: %s\n", cudaGetErrorString(error));
}
