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

static inline std::vector<std::string> _getFiles(const std::string& path)
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

static inline cv::CommandLineParser _getParser(int argc, char** argv)
{
    cv::CommandLineParser parser(
        argc, argv,
        "{mode   m|<none>| Device to run on, GPU or CPU.}"
        "{config c|<none>| Path to JSON config file.}"
        "{folder f|<none>| Path to folder containing ordered frames for detection. First frame will serve as the reference frame when detecting objects.}"

        "{help   h|false | Show help message}"
    );

    parser.about("Usage: ./main --mode=[GPU,CPU] --config=CONFIG_PATH");

    return parser;
}

int main(int argc, char** argv)
{
    int height, width;

    auto parser = _getParser(argc, argv);

    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 1;
    }

    std::string json_config = parser.get<std::string>("config");
    std::string device_mode = parser.get<std::string>("mode");
    std::string folder_path = parser.get<std::string>("folder");

    if (json_config.empty())
    {
        std::cerr << "Path to JSON config file required." << std::endl;
        return -1;
    }

    if (folder_path.empty())
    {
        std::cerr << "Path to folder with frames required." << std::endl;
        return -1;
    }

    std::vector<std::string> files = _getFiles(folder_path);
    std::vector<std::pair<std::string, unsigned char*>> images = _getImages(files, height, width);

    json pipeline_config = _loadJson(json_config);

    if (device_mode == "CPU")
    {
        std::cout << "Running detection pipeline from CPU." << std::endl;
        CPU::runPipeline(images, width, height, pipeline_config);
    }
    else if (device_mode == "GPU")
    {
        std::cout << "Running detection pipeline from GPU." << std::endl;
        GPU::runPipeline(images, width, height, pipeline_config);
    }

    else
    {
        std::cerr << "Invalid mode. --mode=GPU | --mode=CPU" << std::endl;
        return -1;
    }

    for (const auto & image : images)
        delete[] std::get<1>(image);

    std::cout << "Done." << std::endl;

    return 0;
}
