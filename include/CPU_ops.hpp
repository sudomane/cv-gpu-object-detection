#pragma once

#include <tuple>
#include <vector>

typedef std::pair<int,int> t_point;

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