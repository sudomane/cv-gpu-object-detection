#include <pipeline.hpp>

#include <iostream>
#include <chrono>
#include <fstream>
#include <img_ops.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static inline unsigned char* _initRef(unsigned char* ref_image, const t_point& dim)
{
    unsigned char* ref_gray  = new unsigned char[std::get<0>(dim) * std::get<1>(dim)];
    CPU::grayscale(ref_gray, ref_image, dim);

    return ref_gray;
}

static inline int _getMaxLabel(std::vector<std::pair<int,int>> histogram)
{
    int max   = 0;
    int label = 0;

    for (const std::pair<int,int> & e : histogram)
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

static inline void _addToJSON(json& json_data, const std::string& filename, const std::pair<t_point, t_point>& bbox_coords)
{
    int x_min = std::get<0>(std::get<0>(bbox_coords));
    int x_max = std::get<1>(std::get<0>(bbox_coords));

    int y_min = std::get<0>(std::get<1>(bbox_coords));
    int y_max = std::get<1>(std::get<1>(bbox_coords));

    json_data[filename] = {{x_min, y_min} , {x_max, y_max}};
}

static inline void _exportJSON(json& json_data)
{
    std::ofstream outputFile("bbox.json");
    outputFile << json_data;
}

void CPU::runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images, const t_point& dim)
{
    unsigned char* ref_image = _initRef(std::get<1>(images[0]), dim);
    unsigned char* h_buffer  = new unsigned char[width * height];

    int sigma       = 15;
    int kernel_size = 21;
    int bin_thresh  = 8;

    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    json json_data;


    for (int i = 1; i < images.size(); i++)
    {
        const std::string filename = std::get<0>(images[i]);
        unsigned char* image       = std::get<1>(images[i]);

        CPU::grayscale (h_buffer, image, dim);
        CPU::difference(h_buffer, ref_image, dim);
        CPU::gaussian  (h_buffer, dim, kernel_size, sigma);
        CPU::morphology(h_buffer, dim, kernel_size);
        CPU::binary    (h_buffer, dim, bin_thresh);

        auto histogram   = CPU::connectedComponents(h_buffer, dim);
        int  max_label   = _getMaxLabel(histogram);

        const auto bbox_coords = CPU::getBbox(h_buffer, dim, max_label);

        _addToJSON(json_data, filename, bbox_coords);

        std::cout << "[CPU] : " << i << "/" << images.size()-1 << std::endl;
    }

    _exportJSON(json_data);

    delete[] h_buffer;
    delete[] ref_image;
}
