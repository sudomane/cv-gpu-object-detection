#include <nlohmann/json.hpp>
#include <string>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

inline json _loadConfig(const std::string& config_file = "../config.json")
{
    std::ifstream json_file(config_file);

    if (!json_file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        return 1;
    }

    json j;

    try
    {
        json_file >> j;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error parsing JSON file: " << e.what() << '\n';
    }

    json_file.close();

    return j;
}
