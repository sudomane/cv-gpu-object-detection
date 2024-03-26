#include <GPU_ops.cuh>

__device__ void _rowReduce(int* d_labels, int width, int dim) {
    int label = d_labels[dim];
    int min_label = label;

    while (label != 0 && (dim % width) >= 0)
    {
        atomicMin(&min_label, label);

        dim--;
        label = d_labels[dim];
    }
}

__device__ void _colReduce(int* d_labels, int width, int height, int dim) {
    int label = d_labels[dim];
    int min_label = label;

    while (label != 0 && (dim % height) >= 0) {
        atomicMin(&min_label, label);

        dim -= width;
        label = d_labels[dim];
    }
}

__global__ void GPU::initLabelCC(int* d_label, int width, int height)
{
    int dim = blockDim.x * blockIdx.x + threadIdx.x;

    int x = dim % width;
    int y = dim / width;

    if (x>= width || y >= height)
        return;

    int pos = x * height + y;

    d_label[pos] = pos;
}


__global__ void GPU::components(int* d_labels, int width, int height)
{
    int dim = blockDim.x * blockIdx.x + threadIdx.x;

    int x = dim % width;
    int y = dim / width;

    if (x >= width || y >= height)
        return;

    if (d_labels[x * height + y] == 0)
        return;

    _rowReduce(d_labels, width, dim);
    //_colReduce(d_labels, width, height, dim);
}

void GPU::HostWrapper::initLabelCC(int* d_labels)
{
    GPU::initLabelCC<<<this->num_blocks, this->block_size>>>(d_labels, this->width, this->height);

    cudaDeviceSynchronize();

    cudaError_t error = cudaPeekAtLastError();

    if (error != cudaSuccess)
        errx(1, "[initLabelCC] CUDA Error: %s\n", cudaGetErrorString(error));
}

void GPU::HostWrapper::components(int* d_labels)
{
    GPU::components<<<this->num_blocks, this->block_size>>>(d_labels, this->width, this->height);

    cudaDeviceSynchronize();

    cudaError_t error = cudaPeekAtLastError();

    if (error != cudaSuccess)
        errx(1, "[components] CUDA Error: %s\n", cudaGetErrorString(error));
}