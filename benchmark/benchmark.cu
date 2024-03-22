#include <benchmark/benchmark.h>

#include <GPU_ops.cuh>

#include <cmath>

static int width  = 1920;
static int height = 1080;

int block_size = 256;
int num_blocks = (width * height + block_size - 1) / block_size;

static void BM_Grayscale(benchmark::State& state)
{
    unsigned char* h_src = new unsigned char[width * height * 3];
    unsigned char* h_dst = new unsigned char[width * height];

    unsigned char* d_src = _toDevice(h_src, width, height, 3);
    unsigned char* d_dst = _toDevice(h_src, width, height);


    for (auto _ : state)
    {
        GPU::grayscale<<<num_blocks, block_size>>>(d_dst, d_src, width, height);
    }

    cudaDeviceSynchronize(); // Wait for GPU to finish

    delete[] h_src;
    delete[] h_dst;

    cudaFree(d_src);
    cudaFree(d_dst);
}

static void BM_Difference(benchmark::State& state)
{
    unsigned char* h_src = new unsigned char[width * height];
    unsigned char* h_dst = new unsigned char[width * height];

    unsigned char* d_src = _toDevice(h_src, width, height);
    unsigned char* d_dst = _toDevice(h_src, width, height);


    for (auto _ : state)
    {
        GPU::difference<<<num_blocks, block_size>>>(d_dst, d_src, width, height);
    }

    cudaDeviceSynchronize(); // Wait for GPU to finish

    delete[] h_src;
    delete[] h_dst;

    cudaFree(d_src);
    cudaFree(d_dst);
}

static void BM_Gaussian(benchmark::State& state)
{
    int sigma = 10;
    int kernel_size   = 21;
    int kernel_offset = std::floor(kernel_size/2);

    unsigned char* h_src = new unsigned char[width * height];
    unsigned char* h_dst = new unsigned char[width * height];

    unsigned char* d_src = _toDevice(h_src, width, height);
    unsigned char* d_dst = _toDevice(h_src, width, height);

    float* d_kernel = _generateDeviceKernel(kernel_size, sigma);

    for (auto _ : state)
    {
        GPU::gaussian<<<num_blocks, block_size>>>(d_dst, d_src, d_kernel, width, height, kernel_size, kernel_offset);
    }

    cudaDeviceSynchronize(); // Wait for GPU to finish

    delete[] h_src;
    delete[] h_dst;

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_kernel);
}

BENCHMARK(BM_Grayscale) ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Difference)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Gaussian)  ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();