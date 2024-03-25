#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
#include <filesystem>
#include <tuple>
#include <opencv2/opencv.hpp>

#include <pipeline.hpp>

inline json _loadJson(const std::string& config_file)
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

static inline std::vector<std::string> _getFiles(const std::string& path = "../data/rolling_hammer")
{
    std::cout << "Fetching frame data from " << path << std::endl;

    std::vector<std::string> files;

    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (std::filesystem::is_directory(entry))
            continue;

        auto absolute_path = std::filesystem::absolute(entry.path().string());
        files.push_back(absolute_path);
    }

    std::sort(files.begin(), files.end());

    return files;
}

static inline std::vector<std::pair<std::string, unsigned char*>> _getImages(const std::vector<std::string>& files, int& width, int& height)
{
    std::vector<std::pair<std::string, unsigned char*>> images;

    images.reserve(files.size());

    for (const std::string& file : files)
    {
        cv::Mat cv_image = cv::imread(file, cv::IMREAD_COLOR);
        unsigned char* raw_data = new unsigned char[cv_image.cols * cv_image.rows * 3];
        std::memcpy(raw_data, cv_image.data, cv_image.cols * cv_image.rows * 3);

        images.push_back({ file, raw_data });

        width  = cv_image.cols;
        height = cv_image.rows;
    }

    return images;
}

int main(int argc, char** argv)
{
    int height, width;

    cv::CommandLineParser parser(
        argc, argv,
        "{mode   m|<none>| Device to run on, GPU or CPU.}"
        "{config c|<none>| Path to JSON config file.}"

        "{help   h|false | Show help message}"
    );

    parser.about("Example usage: ./main --mode=GPU --config=CONFIG_PATH");

    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 1;
    }

    std::string json_config = parser.get<std::string>("config");
    std::string device_mode = parser.get<std::string>("mode");

    if (json_config.empty())
    {
        std::cerr << "Path to JSON config file required." << std::endl;
        return -1;
    }

    std::vector<std::string> files = _getFiles();
    std::vector<std::pair<std::string, unsigned char*>> images = _getImages(files, height, width);

    std::pair<int, int> dim = { width, height };

    json pipeline_config = _loadJson(json_config);

    if (device_mode == "CPU")
    {
        std::cout << "Running detection pipeline from CPU." << std::endl;
        CPU::runPipeline(images, dim, pipeline_config);
    }
    else if (device_mode == "GPU")
    {
        std::cout << "Running detection pipeline from GPU." << std::endl;
        GPU::runPipeline(images, dim, pipeline_config);
    }

    else
    {
        std::cerr << "Invalid mode." << std::endl;
        return -1;
    }

    for (const auto & image : images)
        delete[] std::get<1>(image);

    std::cout << "Done." << std::endl;

    return 0;
}
