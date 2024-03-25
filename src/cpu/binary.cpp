#include <CPU_ops.hpp>

void CPU::binary(unsigned char* & data, int width, int height, int bin_thresh)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int pos  = i * height + j;
            data[pos] = data[pos] > bin_thresh ? 255 : 0;
        }
    }
}