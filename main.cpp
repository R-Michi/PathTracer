#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "rt_app.h"

int main()
{
    try
    {
        PathTracer::init_static();
        PathTracer ray_tracer;
        ray_tracer.run_app();
    }
    catch (std::exception& e)
    {
        std::cout << "[ERROR]: " << e.what() << std::endl;
    }
    return 0;
}