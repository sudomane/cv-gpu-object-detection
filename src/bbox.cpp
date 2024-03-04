#include <img_ops.hpp>

std::pair<t_point, t_point> cpu::getBbox(unsigned char* & src, const t_point& dim, int label)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    int x_min = width;
    int x_max = 0;

    int y_min = height;
    int y_max = 0;

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int pos = j * width + i;
            int val = src[pos];

            if (val != label)
                continue;

            if (i < x_min)
                x_min = i;
            if (i > x_max)
                x_max = i;

            if (j < y_min)
                y_min = j;
            if (j > y_max)
                y_max = j;
        }
    }

    return {{x_min, y_min}, {x_max, y_max}};
}
