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

inline void _addToJSON(json& json_data, const std::string& filename, const std::pair<std::pair<int,int>, std::pair<int,int>>& bbox_coords)
{
    if (bbox_coords == std::pair<std::pair<int,int>, std::pair<int,int>>())
    {
        json_data[filename] = json::array();
        return;
    }

    int x_min = std::get<0>(std::get<0>(bbox_coords));
    int x_max = std::get<1>(std::get<0>(bbox_coords));

    int y_min = std::get<0>(std::get<1>(bbox_coords));
    int y_max = std::get<1>(std::get<1>(bbox_coords));

    json_data[filename] = {{x_min, y_min} , {x_max, y_max}};
}


namespace CPU
{
    void runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images,
                     int width, int height, const json& config, const std::string& bbox_output);
}; // namespace CPU

namespace GPU
{
    void runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images,
                     int width, int height, const json& config, const std::string& bbox_output);
}; // namespace GPU