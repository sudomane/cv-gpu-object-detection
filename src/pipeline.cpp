#include <pipeline.hpp>

#include <chrono>
#include <CPU_ops.hpp>
#include <opencv2/opencv.hpp>

static inline unsigned char* _initRef(unsigned char* ref_image, int width, int height)
{
    unsigned char* ref_gray  = new unsigned char[width * height];
    CPU::grayscale(ref_gray, ref_image, width, height);

    return ref_gray;
}

static inline int _getMaxLabel(std::vector<t_point> histogram)
{
    int max   = 0;
    int label = 0;

    for (const t_point & e : histogram)
    {
        int n_label = std::get<1>(e);

        if (n_label > max)
        {
            max = n_label;
            label = std::get<0>(e);
        }
    }

    return label;
}

static inline void _saveImage(const unsigned char* image_data, int width, int height, const std::string& filename)
{
    cv::Mat image(width, height, CV_8UC1);
    memcpy(image.data, image_data, width * height * sizeof(unsigned char));

    cv::imwrite(filename, image);
}

void CPU::runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images, int width, int height, const json& config, const std::string& bbox_output)
{
    json bbox_JSON_data;

    int bin_thresh   = config["threshold"];
    int sigma        = config["sigma"];
    int kernel_size  = config["kernel_size"];
    int opening_size = config["opening_size"];
    int closing_size = config["closing_size"];

    unsigned char* ref_image = _initRef(std::get<1>(images[0]), width, height);
    unsigned char* h_buffer  = new unsigned char[width * height];

    for (int i = 1; i < images.size(); i++)
    {
        const std::string filename = std::get<0>(images[i]);
        unsigned char* image       = std::get<1>(images[i]);

        CPU::grayscale (h_buffer, image, width, height);
        CPU::difference(h_buffer, ref_image, width, height);
        CPU::gaussian  (h_buffer, width, height, kernel_size, sigma);
        CPU::morphology(h_buffer, width, height, opening_size, closing_size);
        CPU::binary    (h_buffer, width, height, bin_thresh);

        auto histogram   = CPU::connectedComponents(h_buffer, width, height);
        int  max_label   = _getMaxLabel(histogram);

        std::cout << "Processed frame " << i << " of " << images.size()-1 << std::endl;

        if (max_label == 0)
        {
            _addToJSON(bbox_JSON_data, filename, {});
            continue;
        }

        const auto bbox_coords = CPU::getBbox(h_buffer, width, height, max_label);

        _addToJSON(bbox_JSON_data, filename, bbox_coords);
    }

    _exportJSON(bbox_JSON_data, bbox_output);

    delete[] h_buffer;
    delete[] ref_image;
}