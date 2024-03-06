#include <pipeline.hpp>

void GPU::runPipeline(std::vector<std::pair<std::string, unsigned char*>>& images,
                     const std::pair<int, int> &dim)
{
    for (int i = 0; i < images.size(); i++)
    {
        const std::string filename = std::get<0>(images[i]);
        unsigned char* image       = std::get<1>(images[i]);
    }
}