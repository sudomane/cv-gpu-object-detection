#include <img_ops.hpp>

void cpu::binary(unsigned char* &src, const t_point& dim, int bin_thresh)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int pos = j * width + i;
            src[pos] = src[pos] > bin_thresh ? 100 : 0;
        }
    }
}