#include <pipeline.hpp>

#include <err.h>
#include <GPU_ops.cuh>
#include <opencv2/opencv.hpp>

static inline unsigned char* _cudaMalloc(size_t n)
{
    unsigned char* d_out;

    int rc = cudaMalloc(&d_out, n);
    if (rc)
      errx(1, "Failed to allocate d_out.");

    return d_out;
}

static inline unsigned char* _toDevice(const unsigned char* h_src, int width, int height, int n_channels = 1)
{
    unsigned char* d_dst = _cudaMalloc(sizeof(unsigned char) * width * height * n_channels);

    int rc = cudaMemcpy(d_dst, h_src, sizeof(unsigned char) * width * height * n_channels, cudaMemcpyHostToDevice);
    if (rc)
        errx(1, "Failed to copy host memory to device buffer.");

    return d_dst;
}

static inline void _saveImage(const unsigned char* d_image_data, const t_point& dim, const std::string& filename)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    unsigned char* h_image_data = new unsigned char[width * height];

    int rc = cudaMemcpy(h_image_data, d_image_data, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    if (rc)
        errx(1, "Failed to copy host memory to device buffer while saving image.");

    cv::Mat image(width, height, CV_8UC1);
    memcpy(image.data, h_image_data, width * height * sizeof(unsigned char));

    cv::imwrite(filename, image);

    delete[] h_image_data;
}

static inline unsigned char* _initRef(unsigned char* h_ref_image, const t_point& dim)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    int block_size = 256;
    int num_blocks = (width * height + block_size - 1) / block_size;

    unsigned char* h_ref_gray  = new unsigned char[width * height];

    unsigned char* d_ref_gray  = _cudaMalloc(width * height * sizeof(unsigned char));
    unsigned char* d_ref_image = _toDevice(h_ref_image, width, height, 3);

    GPU::grayscale<<<num_blocks, block_size>>>(d_ref_gray, d_ref_image, width, height);

    delete[] h_ref_gray;

    return d_ref_gray;
}


void GPU::runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images,
                     const std::pair<int, int> &dim)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    int block_size = 256;
    int num_blocks = (width * height + block_size - 1) / block_size;

    unsigned char* d_ref = _initRef(std::get<1>(images[0]), dim);

    for (int i = 1; i < images.size(); i++)
    {
        if (i != 20)
            continue;

        const std::string filename = std::get<0>(images[i]);
        unsigned char* h_image     = std::get<1>(images[i]);
        unsigned char* d_image     = _toDevice(h_image, width, height, 3);
        unsigned char* d_buffer    = _cudaMalloc(width * height * sizeof(unsigned char));

        GPU::grayscale <<<num_blocks, block_size>>>(d_buffer, d_image, width, height);
        GPU::difference<<<num_blocks, block_size>>>(d_buffer, d_ref, width, height);

        _saveImage(d_buffer, dim, "out" + std::to_string(i) + ".png");

        std::cout << "[GPU] : " << i << "/" << images.size()-1 << std::endl;

        cudaFree(d_buffer);
        cudaFree(d_image);
    }

    cudaFree(d_ref);
}