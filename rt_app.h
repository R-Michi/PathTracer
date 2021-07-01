#pragma once

#include <ray_tracing.h>
#include <stb/stb_image_write.h>
#include <random>
#include <atomic>

namespace __internal_random
{
    float uniform_real_dist(float min, float max);
    float normal_dist(float mean, float sigma);
    uint64_t uniform_uint64_dist(uint64_t min, uint64_t max);
};

class RayTracerV2 : private rt::RayTracer
{
private:
    struct RayPayload
    {
        glm::vec3 color;
        bool is_light;
    };

    uint32_t n_threads;

    glm::vec3* environment;
    int env_width, env_height;

    std::atomic_uint32_t rendered_pixel;
    uint64_t* traced_rays;

    static inline float sin_lookup[36000];  // 0.01° resolution
    static inline float cos_lookup[36000];  // 0.01° resolution
    static inline float asin_lookup[9000];

    glm::vec3 ray_generation_shader(float x, float y);
    void closest_hit_shader(const rt::Ray& ray, int recursion, float t, float t_max, const rt::Primitive* hit, void* ray_payload);
    void miss_shader(const rt::Ray& ray, int recursuon, float t_max, void* ray_payload);

    glm::vec3 compute_direction(float x, float y, float aspect, const glm::vec3& dir, const glm::vec3& up, float fov);
    static glm::vec3 vogeldisk_sphere(int sample_index, int samples_count, float phi);
    static glm::vec2 vogeldisk(int sample_index, int samples_count, float phi);
    static float interleaved_gradient_noise(const glm::vec2& pos);

    static glm::vec3 light_sample(const glm::vec2& xi, const glm::vec3& l_direction, float l_distance, float l_radius);
    static glm::vec3 importance_sample_GGX(glm::vec2 Xi, glm::vec3 N, float roughness);
    static float geometry_sub_GGX(const glm::vec3& n, const glm::vec3& x, float k);
    static float geometry_GGX(const glm::vec3& n, const glm::vec3& v, const glm::vec3& l, float a);
    static glm::vec3 fresnel_schlick_roughness(float cos_theta, const glm::vec3& F0, float roughness);

    static inline bool is_light(const rt::Primitive* p);
    static void print_rendered_pixel(std::atomic_bool* should_print, std::atomic_uint32_t* px, uint64_t* traced_rays);

    static void load_lookup(void);
    static inline float fast_sin(float x);
    static inline float fast_cos(float x);
    static inline float fast_asin(float x);

public:
    RayTracerV2(void);
    ~RayTracerV2(void);

    void run_app(void);
};