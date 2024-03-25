#include <CPU_ops.hpp>

#include <cmath>

void CPU::difference(unsigned char* dst, const unsigned char* src, int width, int height)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int pos = i * height + j;

            unsigned char src_val = src[pos];
            unsigned char dst_val = dst[pos];

            unsigned char val = abs(dst_val - src_val);

            dst[pos] = val;
        }
    }
}
