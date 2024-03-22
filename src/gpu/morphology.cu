#include <GPU_ops.cuh>

__device__ void _kernelFunc(unsigned char* d_dst, const unsigned char* d_src, int dim, int width, int height, int kernel_size, int offset, bool is_dilation)
{
    int x = dim % width;
    int y = dim / width;

    int ii_start, ii_end;
    int jj_start, jj_end;

    _getDeviceBoundaries(x, kernel_size, width,  offset, ii_start, ii_end);
    _getDeviceBoundaries(y, kernel_size, height, offset, jj_start, jj_end);

    unsigned char tmp = is_dilation ? 255 : 0;

    for (int ii = ii_start; ii < ii_end; ii++)
    {
        for (int jj = jj_start; jj < jj_end; jj++)
        {
            unsigned char val = d_src[ii * height + jj];

            if (is_dilation)
            {
                if (val < tmp)
                    tmp = val;
            }

            else
            {
                if (val > tmp)
                    tmp = val;
            }

        }
    }

    d_dst[x * height + y] = tmp;
}

__global__ void GPU::morphology (unsigned char* d_dst, unsigned char* d_src, int width, int height, int opening_size, int closing_size, int offset)
{
    int dim = blockDim.x * blockIdx.x + threadIdx.x;

    int x = dim % width;
    int y = dim / width;

    if (x >= width || y >= height)
        return;

    // FIXME

    // Closing
    {
        _kernelFunc(d_dst, d_src, dim, width, height, opening_size, offset, true);  // Dilation
        _kernelFunc(d_src, d_dst, dim, width, height, opening_size, offset, false); // Erosion
    }

    // Opening
    {
        //_kernelFunc(d_dst, d_src, dim, width, height, opening_size, offset, false); // Erosion
        //_kernelFunc(d_src, d_dst, dim, width, height, opening_size, offset, true);  // Dilation
    }
}
