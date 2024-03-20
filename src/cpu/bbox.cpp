#include <CPU_ops.hpp>

#include <algorithm>
#include <iostream>

std::pair<t_point, t_point> CPU::getBbox(unsigned char* & src, const t_point& dim, int label)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    int top = height;
    int bot = 0;

    int left = width;
    int right = 0;

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int pos = i * height + j;
            int val = src[pos];

            if (val != label)
                continue;

            top   = std::min(top,  j);
            left  = std::min(left, i);

            bot   = std::max(bot,   j);
            right = std::max(right, i);
        }
    }

    return {{top, left}, {bot, right}};
}
