#include <CPU_ops.hpp>

#include <algorithm>
#include <cmath>
#include <functional>

#include <iostream>

static std::tuple<int, int> getBoundaries(int i, int kernel_size, int size)
{
    int start = 0;
    int end   = size;

    int kernel_size_ = std::floor(kernel_size/2);

    if (i > kernel_size_)
        start = i - kernel_size_;

    if (i + kernel_size_ < size)
        end = i + kernel_size_ + 1;

    return std::make_tuple(start, end);
}

static void kernelFunc(unsigned char* & src, const t_point &dim, int kernel_size, bool is_dilation)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    unsigned char * dst = new unsigned char[width * height];

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            std::tuple<int, int> ii_dim = getBoundaries(i, kernel_size, width);
            std::tuple<int, int> jj_dim = getBoundaries(j, kernel_size, height);

            std::vector<unsigned char> kernel;
            kernel.reserve(kernel_size * kernel_size);

            for (int ii = std::get<0>(ii_dim); ii < std::get<1>(ii_dim); ii++)
            {
                for (int jj = std::get<0>(jj_dim); jj < std::get<1>(jj_dim); jj++)
                {
                    unsigned char val = src[jj * width + ii];

                    kernel.push_back(val);
                }
            }
            unsigned char element;

            if (is_dilation)
                element = *(std::max_element(kernel.begin(), kernel.end()));
            else
                element = *(std::min_element(kernel.begin(), kernel.end()));

            dst[j * width + i] = element;
        }
    }

    std::swap(src, dst);

    delete[] dst;
}

void CPU::morphology(unsigned char* & src, const t_point & dim, int kernel_size)
{
    // Closing
    {
        kernelFunc(src, dim, kernel_size, true);  // Dilation
        kernelFunc(src, dim, kernel_size, false); // Erosion
    }

    // Opening
    {
        kernelFunc(src, dim, kernel_size, false); // Erosion
        kernelFunc(src, dim, kernel_size, true);  // Dilation
    }
}
