#include <algorithm>
#include <iostream>
#include <memory>
#include <filesystem>
#include <tuple>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>

#include <pipeline.hpp>

std::vector<std::string> getFiles(const std::string& path = "../data")
{
    std::vector<std::string> files;

    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        auto absolute_path = std::filesystem::absolute(entry.path().string());
        files.push_back(absolute_path);
    }

    std::sort(files.begin(), files.end());

    return files;
}

int main(int argc, char** argv)
{
    std::vector<std::string> files = getFiles();

    std::vector<std::pair<std::string, unsigned char*>> images;

    images.reserve(files.size());

    int height, width;

    for (const std::string& file : files)
    {
        cv::Mat cv_image = cv::imread(file, cv::IMREAD_COLOR);
        unsigned char* raw_data = new unsigned char[cv_image.cols * cv_image.rows * 3];
        std::memcpy(raw_data, cv_image.data, cv_image.cols * cv_image.rows * 3);

        images.push_back({ file, raw_data });

        width  = cv_image.cols;
        height = cv_image.rows;
    }

    std::pair<int, int> dim = {width, height};

    CPU::runPipeline(images, dim);

    for (const auto & image : images)
        delete[] std::get<1>(image);

    return 0;
}
