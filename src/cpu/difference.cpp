#include <CPU_ops.hpp>

#include <cmath>

void CPU::difference(unsigned char* dst, const unsigned char* src, const t_point& dim)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            unsigned char src_val = src[j * width + i];
            unsigned char dst_val = dst[j * width + i];

            unsigned char val = abs(dst_val - src_val);

            dst[j * width + i] = val;
        }
    }
}
