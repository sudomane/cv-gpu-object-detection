#include <CPU_ops.hpp>

void CPU::grayscale(unsigned char* dst, const unsigned char* src, const t_point& dim)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            unsigned char r, g, b;

            int pos     = i * height + j;
            int rgb_pos = 3 * pos;

            r = src[rgb_pos];
            g = src[rgb_pos + 1];
            b = src[rgb_pos + 2];

            unsigned char gray = static_cast<unsigned char>((r + g + b) / 3);

            dst[pos] = gray;
        }
    }
}