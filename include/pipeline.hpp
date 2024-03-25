#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

inline void _exportJSON(json& json_data, const std::string& output = "bbox.json")
{
    std::cout << "Exporting JSON with bounding box data to: " << output << std::endl;
    std::ofstream outputFile(output);
    outputFile << json_data;
}

namespace CPU
{
    void runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images,
                     int width, int height, const json& config);
}; // namespace CPU

namespace GPU
{
    void runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images,
                     int width, int height, const json& config);
}; // namespace GPU