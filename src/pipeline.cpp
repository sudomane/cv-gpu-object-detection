#include <pipeline.hpp>

#include <chrono>
#include <fstream>
#include <img_ops.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static inline unsigned char* _initRef(unsigned char* ref_image, const t_point& dim)
{
    unsigned char* ref_gray  = new unsigned char[std::get<0>(dim) * std::get<1>(dim)];
    cpu::grayscale(ref_gray, ref_image, dim);

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

static inline void _saveToJSON(json& json_data, const std::string& filename, const std::pair<t_point, t_point>& bbox_coords)
{
    int x_min = std::get<0>(std::get<0>(bbox_coords));
    int x_max = std::get<1>(std::get<0>(bbox_coords));

    int y_min = std::get<0>(std::get<1>(bbox_coords));
    int y_max = std::get<1>(std::get<1>(bbox_coords));

    json_data[filename] = {{x_min, y_min} , {x_max, y_max}};
}

static inline void _exportJSON(json& json_data)
{
    std::ofstream outputFile("output.json");
}

void cpu::runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images, unsigned char* &h_buffer, const t_point& dim)
{
    unsigned char* ref_image = _initRef(std::get<1>(images[0]), dim);

    int sigma       = 15;
    int kernel_size = 17;
    int bin_thresh  = 8;

    // auto start_time = std::chrono::system_clock::now();
    json json_data;

    for (int i = 1; i < images.size(); i++)
    {
        const std::string filename = std::get<0>(images[i]);
        unsigned char* image       = std::get<1>(images[i]);

        cpu::grayscale (h_buffer, image, dim);
        cpu::difference(h_buffer, ref_image, dim);
        cpu::gaussian  (h_buffer, dim, kernel_size, sigma);
        cpu::morphology(h_buffer, dim, kernel_size);
        cpu::binary    (h_buffer, dim, bin_thresh);

        auto histogram   = cpu::connectedComponents(h_buffer, dim);
        int  max_label   = _getMaxLabel(histogram);

        const auto bbox_coords = cpu::getBbox(h_buffer, dim, max_label);

        _saveToJSON(json_data, filename, bbox_coords);
        break;
    }

    _exportJSON(json_data);

    //auto end_time = std::chrono::system_clock::now();
    //auto delta = end_time - start_time;
    //float fps = (images.size() - 1) / delta;

    delete[] ref_image;
}
