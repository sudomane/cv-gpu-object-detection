#include <CPU_ops.hpp>

#include <algorithm>
#include <functional>

#include <iostream>

static inline void _kernelFunc(unsigned char* & src, int width, int height, int kernel_size, bool is_dilation)
{
    unsigned char * dst = new unsigned char[width * height];

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            std::tuple<int, int> ii_dim = _getBoundaries(i, kernel_size, width);
            std::tuple<int, int> jj_dim = _getBoundaries(j, kernel_size, height);

            std::vector<unsigned char> kernel;
            kernel.reserve(kernel_size * kernel_size);

            for (int ii = std::get<0>(ii_dim); ii < std::get<1>(ii_dim); ii++)
            {
                for (int jj = std::get<0>(jj_dim); jj < std::get<1>(jj_dim); jj++)
                {
                    unsigned char val = src[ii * height + jj];

                    kernel.push_back(val);
                }
            }
            unsigned char element;

            if (is_dilation)
                element = *(std::max_element(kernel.begin(), kernel.end()));
            else
                element = *(std::min_element(kernel.begin(), kernel.end()));

            dst[i * height + j] = element;
        }
    }

    std::swap(src, dst);

    delete[] dst;
}

void CPU::morphology(unsigned char* & src, int width, int height, int opening_size, int closing_size)
{
    // Closing
    {
        _kernelFunc(src, width, height, closing_size, true);  // Dilation
        _kernelFunc(src, width, height, closing_size, false); // Erosion
    }

    // Opening
    {
        _kernelFunc(src, width, height, opening_size, false); // Erosion
        _kernelFunc(src, width, height, opening_size, true);  // Dilation
    }
}
