#pragma once

#include <tuple>
#include <vector>
#include <string>

namespace cpu
{
    void runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images,
                     const std::pair<int, int> &dim);
};