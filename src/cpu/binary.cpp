#include <CPU_ops.hpp>

void CPU::binary(unsigned char* &src, const t_point& dim, int bin_thresh)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int pos  = i * height + j;
            src[pos] = src[pos] > bin_thresh ? 255 : 0;
        }
    }
}