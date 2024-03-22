#include <benchmark/benchmark.h>

#include <GPU_ops.cuh>

#include <cmath>

static int width  = 1920;
static int height = 1080;

int block_size = 256;
int num_blocks = (width * height + block_size - 1) / block_size;

static void BM_Grayscale(benchmark::State& state)
{
    unsigned char* d_src = _cudaMalloc<unsigned char>(width * height * 3);
    unsigned char* d_dst = _cudaMalloc<unsigned char>(width * height);

    for (auto _ : state)
    {
        GPU::grayscale<<<num_blocks, block_size>>>(d_dst, d_src, width, height);
    }

    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaFree(d_src);
    cudaFree(d_dst);
}

static void BM_Difference(benchmark::State& state)
{
    unsigned char* d_dst = _cudaMalloc<unsigned char>(width * height);
    unsigned char* d_src = _cudaMalloc<unsigned char>(width * height);

    for (auto _ : state)
    {
        GPU::difference<<<num_blocks, block_size>>>(d_dst, d_src, width, height);
    }

    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaFree(d_src);
    cudaFree(d_dst);
}

static void BM_Gaussian(benchmark::State& state)
{
    int sigma = 10;
    int kernel_size   = 21;
    int kernel_offset = std::floor(kernel_size/2);

    unsigned char* d_src = _cudaMalloc<unsigned char>(width * height);
    unsigned char* d_dst = _cudaMalloc<unsigned char>(width * height);

    float* d_kernel = _generateDeviceKernel(kernel_size, sigma);

    for (auto _ : state)
    {
        GPU::gaussian<<<num_blocks, block_size>>>(d_dst, d_src, d_kernel, width, height, kernel_size, kernel_offset);
    }

    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_kernel);
}

static void BM_Binary(benchmark::State& state)
{
    int threshold = 21;

    unsigned char* d_src = _cudaMalloc<unsigned char>(width * height);

    for (auto _ : state)
    {
        GPU::binary<<<num_blocks, block_size>>>(d_src, threshold, width, height);
    }

    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaFree(d_src);
}

static void BM_Morphology(benchmark::State& state)
{
    int opening_size = 21;
    int closing_size = 21;
    int opening_offset = std::floor(opening_size / 2);
    int closing_offset = std::floor(closing_size / 2);

    unsigned char* d_src = _cudaMalloc<unsigned char>(width * height);
    unsigned char* d_dst = _cudaMalloc<unsigned char>(width * height);
    unsigned char* d_buf = _cudaMalloc<unsigned char>(width * height);

    for (auto _ : state)
    {
        GPU::morphology<<<num_blocks, block_size>>>(d_dst, d_src, d_buf, width, height, opening_size, closing_size, opening_offset, closing_offset);
    }

    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_buf);
}

//BENCHMARK(BM_Grayscale) ->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK(BM_Difference)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK(BM_Gaussian)  ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Morphology)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK(BM_Binary)    ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();