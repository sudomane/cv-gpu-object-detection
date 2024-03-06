#pragma once

#include <tuple>
#include <vector>
#include <string>

namespace CPU
{
    void runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images,
                     const std::pair<int, int> &dim);
}; // namespace CPU

namespace GPU
{
    void runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images,
                     const std::pair<int, int> &dim);
}; // namespace GPU