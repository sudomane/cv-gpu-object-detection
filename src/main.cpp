#include <algorithm>
#include <iostream>
#include <memory>
#include <filesystem>
#include <tuple>
#include <opencv2/opencv.hpp>

#include <pipeline.hpp>

static inline std::vector<std::string> _getFiles(const std::string& path = "../data/rolling_hammer")
{
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

    std::vector<std::string> files = _getFiles();
    std::vector<std::pair<std::string, unsigned char*>> images = _getImages(files, height, width);

    std::pair<int, int> dim = { width, height };

    //CPU::runPipeline(images, dim);
    GPU::runPipeline(images, dim);

    for (const auto & image : images)
        delete[] std::get<1>(image);

    return 0;
}
