#include <CPU_ops.hpp>

#include <iostream>
#include <stack>

static inline bool _checkBoundaries(int coord, int limit)
{
    return (coord >= limit || coord < 0);
}

static inline void _fillLabel(unsigned char* & src, t_point dim, int x, int y, unsigned char label, int* count)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    std::stack<t_point> stack;

    stack.push( {x, y} );

    while (!stack.empty())
    {
        t_point position = stack.top();
        stack.pop();

        x = std::get<0>(position);
        y = std::get<1>(position);

        if (_checkBoundaries(x, width) || _checkBoundaries(y, height))
            continue;

        int pos  = x * height + y;
        unsigned char val = src[pos];

        if (val <= label)
            continue;

        src[pos] = label;

        (*count)++;

        stack.push( {x+1, y} );
        stack.push( {x, y+1} );
        stack.push( {x-1, y} );
        stack.push( {x, y-1} );

        stack.push( {x+1, y+1} );
        stack.push( {x-1, y+1} );
        stack.push( {x+1, y-1} );
        stack.push( {x-1, y-1} );
    }
}

std::vector<std::pair<int,int>> CPU::connectedComponents(unsigned char* &src, const t_point& dim)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    unsigned char label = 0;

    std::vector<std::pair<int, int>> label_histogram;

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            if (src[i * height + j] <= label)
                continue;

            label++;

            int n_label = 0;

            _fillLabel(src, dim, i, j, label, &n_label);

            label_histogram.push_back( {label, n_label} );
        }
    }

    return label_histogram;
}
