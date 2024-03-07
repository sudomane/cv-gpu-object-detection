#pragma once

#include <tuple>
#include <vector>

typedef std::pair<int,int> t_point;

namespace GPU
{
    __global__ void grayscale ();
}; // namespace cpu