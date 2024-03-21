#include <GPU_ops.cuh>

__device__ void _getBoundaries(int i, int kernel_size, int size, int offset, int& start, int& end)
{
    int _start = 0;
    int _end   = size;

    if (i > offset)
        _start = i - offset;

    if (i + offset < size)
        _end = i + offset + 1;

    start = _start;
    end   = _end;
}

__global__ void GPU::gaussian(unsigned char* d_dst, const unsigned char* d_src, const float* d_kernel, int width, int height, int kernel_size, int sigma, int offset)
{
    int dim = blockDim.x * blockIdx.x + threadIdx.x;

    int x = dim % width;
    int y = dim / width;

    if (x >= width || y >= height)
        return;

    int ii_start, ii_end;
    int jj_start, jj_end;

    _getBoundaries(x, kernel_size, width,  offset, ii_start, ii_end);
    _getBoundaries(y, kernel_size, height, offset, jj_start, jj_end);

    float val = 0;

    for (int ii = ii_start; ii < ii_end; ii++)
    {
        for (int jj = jj_start; jj < jj_end; jj++)
        {
            int i_kernel = ii - x + offset;
            int j_kernel = jj - y + offset;

            float k_val = d_kernel[i_kernel * kernel_size + j_kernel];
            float src_val = static_cast<float>(d_src[ii * height + jj]);

            val += src_val * k_val;
        }
    }

    d_dst[x * height + y] = static_cast<unsigned char>(val);
}
