#include <CPU_ops.hpp>

#include <memory>

static inline float* _generateKernel(int kernel_size, float sigma)
{
    float  sum = 0;
    float* kernel = new float[kernel_size * kernel_size];

    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            float x = i - kernel_size / 2;
            float y = j - kernel_size / 2;

            float val = std::exp2f(-(x*x + y*y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[j * kernel_size + i] = val;

            sum += val;
        }
    }

    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            kernel[j * kernel_size + i] /= sum;
        }
    }

    return kernel;
}
void CPU::gaussian(unsigned char* & src, int width, int height, int kernel_size, float sigma)
{
    int offset = std::floor(kernel_size / 2);

    float* kernel      = _generateKernel(kernel_size, sigma);
    unsigned char* dst = new unsigned char[width * height];

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            float val = 0;

            std::tuple<int, int> ii_dim = _getBoundaries(i, kernel_size, width);
            std::tuple<int, int> jj_dim = _getBoundaries(j, kernel_size, height);

            for (int ii = std::get<0>(ii_dim); ii < std::get<1>(ii_dim); ii++)
            {
                for (int jj = std::get<0>(jj_dim); jj < std::get<1>(jj_dim); jj++)
                {
                    int i_kernel = ii - i + offset;
                    int j_kernel = jj - j + offset;

                    float k_val   = kernel[i_kernel * kernel_size + j_kernel];
                    float src_val = static_cast<float>(src[ii * height + jj]);

                    val += src_val * k_val;
                }
            }

            dst[i * height + j] = static_cast<unsigned char>(val);
        }
    }

    std::swap(src, dst);

    delete[] dst;
    delete[] kernel;
}
