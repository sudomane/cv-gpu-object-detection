#pragma once

#include <cmath>
#include <tuple>
#include <vector>

typedef std::pair<int,int> t_point;

inline std::tuple<int, int> _getBoundaries(int i, int kernel_size, int size)
{
    int start = 0;
    int end   = size;

    int kernel_size_ = std::floor(kernel_size/2);

    if (i > kernel_size_)
        start = i - kernel_size_;

    if (i + kernel_size_ < size)
        end = i + kernel_size_ + 1;

    return std::make_tuple(start, end);
}

namespace CPU
{
    void grayscale (unsigned char* dst,  const unsigned char* src, const t_point& dim);
    void difference(unsigned char* dst,  const unsigned char* src, const t_point& dim);
    void gaussian  (unsigned char* & src, const t_point& dim, int kernel_size, float sigma);
    void morphology(unsigned char* & src, const t_point& dim, int opening_size, int closing_size);
    void binary    (unsigned char* & src, const t_point& dim, int bin_thresh);

    std::vector<std::pair<int,int>> connectedComponents(unsigned char* & src, const t_point& dim);

    std::pair<t_point, t_point> getBbox(unsigned char* & src, const t_point& dim, int label);

}; // namespace cpu