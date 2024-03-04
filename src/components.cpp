#include <img_ops.hpp>

#include <iostream>

static inline bool _checkBoundaries(int coord, int limit)
{
    return (coord >= limit || coord < 0);
}

static inline void _fillLabel(unsigned char* & src, const t_point& dim, int x, int y, unsigned char label, int* count)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    if (_checkBoundaries(x, width) || _checkBoundaries(y, height))
        return;

    unsigned char val = src[y * width + x];

    if (val <= label)
        return;

    src[y * width + x] = label;

    (*count)++;

    _fillLabel(src, dim, x + 1, y, label, count);
    _fillLabel(src, dim, x, y + 1, label, count);
    _fillLabel(src, dim, x - 1, y, label, count);
    _fillLabel(src, dim, x, y - 1, label, count);
}

std::vector<std::pair<int,int>> cpu::connectedComponents(unsigned char* &src, const t_point& dim)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    unsigned char label = 0;

    std::vector<std::pair<int, int>> label_histogram;

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            if (src[j * width + i] <= label)
                continue;

            label++;

            int n_label = 0;

            _fillLabel(src, dim, i, j, label, &n_label);

            label_histogram.push_back( {label, n_label} );
        }
    }

    return label_histogram;
}
