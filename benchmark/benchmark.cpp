#include <benchmark/benchmark.h>

#include <CPU_ops.hpp>
#include <utils.hpp>

json config = _loadConfig();

static int width  = 1920;
static int height = 1080;

static void BM_Grayscale(benchmark::State& state)
{
    unsigned char* src = new unsigned char[width * height * 3];
    unsigned char* dst = new unsigned char[width * height];

    for (auto _ : state)
    {
        CPU::grayscale(dst, src, width, height);
    }

    delete[] src;
    delete[] dst;
}

static void BM_Difference(benchmark::State& state)
{
    unsigned char* src = new unsigned char[width * height];
    unsigned char* dst = new unsigned char[width * height];

    for (auto _ : state)
    {
        CPU::difference(dst, src, width, height);
    }

    delete[] src;
    delete[] dst;
}

static void BM_Gaussian(benchmark::State& state)
{
    int sigma = config["sigma"];
    int kernel_size = config["kernel_size"];

    unsigned char* src = new unsigned char[width * height];

    for (auto _ : state)
    {
        CPU::gaussian(src, width, height, kernel_size, sigma);
    }

    delete[] src;
}

static void BM_Morphology(benchmark::State& state)
{
    int opening_size = config["opening_size"];
    int closing_size = config["closing_size"];

    unsigned char* src = new unsigned char[width * height];

    for (auto _ : state)
    {
        CPU::morphology(src, width, height, opening_size, closing_size);
    }

    delete[] src;
}

static void BM_Binary(benchmark::State& state)
{
    int threshold = config["threshold"];

    unsigned char* src = new unsigned char[width * height];

    for (auto _ : state)
    {
        CPU::binary(src, width, height, threshold);
    }

    delete[] src;
}

static inline void _populateImage(unsigned char* src)
{
    // Populate image with synthetic objects
    for (int i = 500; i < 1800; i++)
    {
        for (int j = 60; j < 700; j++)
        {
            src[i * height + j] = 255;
        }
    }

    for (int i = 200; i < 400; i++)
    {
        for (int j = 60; j < 1000; j++)
        {
            src[i * height + j] = 255;
        }
    }

    for (int i = 0; i < 150; i++)
    {
        for (int j = 0; j < 1000; j++)
        {
            src[i * height + j] = 255;
        }
    }
}

static void BM_Components(benchmark::State& state)
{
    unsigned char* src = static_cast<unsigned char*>(calloc(width * height, sizeof(unsigned char)));

    _populateImage(src);

    for (auto _ : state)
    {
        CPU::connectedComponents(src, width, height);
    }

    delete[] src;
}

static void BM_BBox(benchmark::State& state)
{
    unsigned char* src = static_cast<unsigned char*>(calloc(width * height, sizeof(unsigned char)));

    // Populate image with synthetic object
    for (int i = 500; i < 1800; i++)
    {
        for (int j = 60; j < 700; j++)
        {
            src[i * height + j] = 1;
        }
    }

    for (auto _ : state)
    {
        CPU::getBbox(src, width, height, 1);
    }

    delete[] src;
}

BENCHMARK(BM_Grayscale) ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Difference)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Gaussian)  ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Morphology)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Binary)    ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Components)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BBox)      ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();