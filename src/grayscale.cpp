#include <img_ops.hpp>

void cpu::grayscale(unsigned char* dst, const unsigned char* src, const t_point& dim)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            unsigned char r, g, b;

            int pos = 3 * (j * width + i);

            r = src[pos];
            g = src[pos + 1];
            b = src[pos + 2];

            unsigned char gray = static_cast<unsigned char>((r + g + b) / 3);

            dst[j * width + i] = gray;
        }
    }
}