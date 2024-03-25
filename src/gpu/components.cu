#include <GPU_ops.cuh>

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

__global__ void GPU::components(unsigned char* d_data, int* d_labels, int width, int height)
{

}