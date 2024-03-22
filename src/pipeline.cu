#include <pipeline.hpp>

#include <cmath>
#include <GPU_ops.cuh>
#include <opencv2/opencv.hpp>

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

    unsigned char* d_ref_gray  = _cudaMalloc<unsigned char>(width * height);
    unsigned char* d_ref_image = _toDevice<unsigned char>(h_ref_image, width, height, 3);

    GPU::grayscale<<<num_blocks, block_size>>>(d_ref_gray, d_ref_image, width, height);

    delete[] h_ref_gray;

    return d_ref_gray;
}

void GPU::runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images,
                     const std::pair<int, int> &dim)
{
    int width  = std::get<0>(dim);
    int height = std::get<1>(dim);

    int threshold     = 60; // TODO: Implement adaptative thresholding
    int sigma         = 10;
    int kernel_size   = 21;
    int opening_size  = 21;
    int closing_size  = 21;
    int kernel_offset = std::floor(kernel_size / 2);

    int block_size = 256;
    int num_blocks = (width * height + block_size - 1) / block_size;

    float* d_kernel             = _generateDeviceKernel(kernel_size, sigma);
    unsigned char* d_ref        = _initRef(std::get<1>(images[0]), dim);
    unsigned char* d_buffer     = _cudaMalloc<unsigned char>(width * height);
    unsigned char* d_buffer_tmp = _cudaMalloc<unsigned char>(width * height);

    for (int i = 1; i < images.size(); i++)
    {
        const std::string filename = std::get<0>(images[i]);
        unsigned char* h_image     = std::get<1>(images[i]);
        unsigned char* d_image     = _toDevice<unsigned char>(h_image, width, height, 3);

        GPU::grayscale <<<num_blocks, block_size>>>(d_buffer, d_image, width, height);
        GPU::difference<<<num_blocks, block_size>>>(d_buffer, d_ref, width, height);
        GPU::gaussian  <<<num_blocks, block_size>>>(d_buffer_tmp, d_buffer, d_kernel, width, height, kernel_size, kernel_offset);
        GPU::morphology<<<num_blocks, block_size>>>(d_buffer, d_buffer_tmp, width, height, opening_size, closing_size, kernel_offset);
        GPU::binary    <<<num_blocks, block_size>>>(d_buffer_tmp, threshold, width, height);

        _saveImage(d_buffer_tmp, dim, "out" + std::to_string(i) + ".png");

        std::cout << "[GPU] : " << i << "/" << images.size()-1 << std::endl;

        cudaFree(d_image);
    }

    cudaFree(d_buffer);
    cudaFree(d_buffer_tmp);
    cudaFree(d_ref);
    cudaFree(d_kernel);
}
