#include "rt_app.h"
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <thread>
#include <glm/gtc/matrix_transform.hpp>

#define SCR_WIDTH       (960 * 1)
#define SCR_HEIGHT      (540 * 1)
#define SCR_ASPECT      ((float)SCR_WIDTH / (float)SCR_HEIGHT)
#define N_PIXELS        (SCR_WIDTH * SCR_HEIGHT)
#define N_PIXELS_100    (N_PIXELS / 100)
#define PRIM_COUNT      3

#ifndef M_PI_2
    #define M_PI_2_F 1.5707963f
#endif
#ifndef M_2PI
    #define M_2_PI_F 6.2831853f
#endif

std::random_device rd;
std::seed_seq seed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
std::mt19937 rand_engine(seed);

float __internal_random::uniform_real_dist(float min, float max)
{
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rand_engine);
}

float __internal_random::normal_dist(float mean, float sigma)
{
    std::normal_distribution<float> dist(mean, sigma);
    return dist(rand_engine);
}

uint64_t __internal_random::uniform_uint64_dist(uint64_t min, uint64_t max)
{
    std::uniform_int_distribution<uint64_t> dist(min, max);
    return dist(rand_engine);
}

float RadicalInverse_VdC(uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10; // / 0x100000000
}

// ----------------------------------------------------------------------------
glm::vec2 Hammersley(uint32_t i, uint32_t N)
{
    return glm::vec2((float)i / (float)N, RadicalInverse_VdC(i));
}

RayTracerV2::RayTracerV2(void)
{
    rt::Framebuffer fbo;
    fbo.width = SCR_WIDTH;
    fbo.height = SCR_HEIGHT;
    this->set_framebuffer(fbo);

    this->environment = (glm::vec3*)stbi_loadf("../../../assets/skyboxes/environment.hdr", &this->env_width, &this->env_height, nullptr, 3);
    if(this->environment == nullptr)
        throw std::runtime_error("Failed to load environment.");

    rt::Sphere spheres[PRIM_COUNT] = {
        rt::Sphere(
            glm::vec3(0.0f, 0.0f, 3.5f),
            1.0f,
            {
                glm::vec3(1.0f, 1.0f, 1.0f),
                0.3f,
                0.0f,
                1.0f
            }
        ),
        rt::Sphere(
            glm::vec3(0.0f, 0.0f, 6.0f),
            0.2f,
            {
                glm::vec3(100.0f, 100.0f, 100.0f),
                -1.0f,
                0.0f,
                1.0f
            }
        ),
        rt::Sphere(
            glm::vec3(0.0f, -10001.0f, 0.0f),
            10000.0f,
            {
                glm::vec3(0.0f, 0.0f, 1.0f),
                0.5f,
                0.5f,
                1.0f
            }
        )
    };

    rt::BufferLayout buffer_layout = {};
    buffer_layout.size = PRIM_COUNT;
    buffer_layout.first = 0;
    buffer_layout.last = buffer_layout.size;

    rt::Buffer primitive_buffer(buffer_layout);
    primitive_buffer.data(0, 3, spheres);

    this->draw_buffer(primitive_buffer);

    this->n_threads = omp_get_max_threads() * 4;
    this->set_num_threads(n_threads);

    this->rendered_pixel = 0;
    this->traced_rays = new uint64_t[this->n_threads];
    memset(this->traced_rays, 0, sizeof(uint64_t) * this->n_threads);

    RayTracerV2::load_lookup();
}

RayTracerV2::~RayTracerV2(void)
{
     const rt::Framebuffer& fbo = this->get_framebuffer();
     stbi_write_png("../../../output.png", fbo.width, fbo.height, 3, fbo.buff, fbo.width * sizeof(rt::Color3ui8));

     stbi_image_free(this->environment);
     delete[] this->traced_rays;
}

glm::vec3 RayTracerV2::ray_generation_shader(float x, float y)
{
    const glm::vec3 cam_pos = glm::vec3(5.0f, 0.0f, 10.0f);
    const glm::vec3 look_at(0.0f, 0.0f, 3.5f);
    const glm::vec3 up(0.0f, 1.0f, 0.0f);

    rt::Ray ray;
    ray.origin = cam_pos;
    ray.direction = this->compute_direction(x, y, SCR_ASPECT, look_at - cam_pos, up, glm::radians(100.0f));

    RayPayload payload;
    this->trace_ray(ray, 2, 100.0f, &payload);
    ++this->rendered_pixel;
    ++this->traced_rays[omp_get_thread_num()];

    glm::vec3 LDR = payload.color / (payload.color + 1.0f);
    return glm::pow(LDR, glm::vec3(1.0f / 2.2f));
}

void RayTracerV2::closest_hit_shader(const rt::Ray& ray, int recursion, float t, float t_max, const rt::Primitive* hit, void* ray_payload)
{
    using namespace __internal_random;
    
    if (is_light(hit))
    {
        ((RayPayload*)ray_payload)->color = hit->material().albedo;
        ((RayPayload*)ray_payload)->is_light = true;
        return;
    }
    if (recursion == 1)
    {
        ((RayPayload*)ray_payload)->color = glm::vec3(0.0f);
        ((RayPayload*)ray_payload)->is_light = false;
        return;
    }

    // get hit-primitive type
    const rt::Sphere* hit_sphere = dynamic_cast<const rt::Sphere*>(hit);

    // get properties of the intersection point
    const glm::vec3 intersection = ray.origin + t * ray.direction;
    const glm::vec3 normal = glm::normalize(intersection - hit_sphere->center());
    const glm::vec3 view = -ray.direction;

    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::mat3 TBN;
    TBN[2] = glm::normalize(normal);
    TBN[0] = glm::normalize(glm::cross(up, TBN[2]));
    TBN[1] = cross(TBN[2], TBN[0]);

    const float vogel_rotation = uniform_real_dist(0.0f, M_2_PI_F);

    // direct light
    constexpr int n_direct_samples = 1;
    glm::vec3 direct_light(0.0f);

    for (int i = 0; i < 1; ++i)
    {
        const glm::vec3 light_pos(0.0f, 0.0f, 6.0f);
        const glm::vec3 light_direction = glm::normalize(light_pos - intersection);
        const float light_radius = 0.2f;
        const float light_distance = glm::distance(intersection, light_pos);
        glm::vec3 irradiance_light(0.0f);

        for (int j = 0; j < n_direct_samples; ++j)
        {
            const glm::vec2 xi = Hammersley(j, n_direct_samples);

            rt::Ray sample_ray;
            sample_ray.direction = light_sample(xi, light_direction, light_distance, light_radius);
            sample_ray.origin = intersection;

            if (glm::dot(normal, sample_ray.direction) > 0.0f)
            {
                RayPayload payload;
                this->trace_ray(sample_ray, 1, t_max, &payload);
                ++this->traced_rays[omp_get_thread_num()];

                rt::Light sample_light;
                sample_light.direction = sample_ray.direction;
                sample_light.intensity = payload.color;

                irradiance_light += rt::light(sample_light, hit->material(), view, normal);
            }
        }
        direct_light += (irradiance_light / (float)n_direct_samples);
    }

    // indirect diffuse
    constexpr int n_diffuse_samples = 1000;
    glm::vec3 irradiance_diffuse(0.0f);

    for (int i = 0; i < n_diffuse_samples; ++i)
    {
        rt::Ray sample_ray;
        sample_ray.origin = intersection;
        sample_ray.direction = TBN * glm::normalize(vogeldisk_sphere(i, n_diffuse_samples, vogel_rotation));

        RayPayload payload;
        this->trace_ray(sample_ray, recursion - 1, t_max, &payload);
        ++this->traced_rays[omp_get_thread_num()];

        const float NdotL = glm::max(glm::dot(normal, sample_ray.direction), 0.0f);
        irradiance_diffuse += payload.is_light ? glm::vec3(0.0f) : payload.color * NdotL;
    }
    irradiance_diffuse = (hit->material().albedo * (float)M_PI / (float)n_diffuse_samples) * irradiance_diffuse;

    // indirect specular light
    constexpr float n_specular_samples = 10;
    float total_weight = 0.0f;
    glm::vec3 irradiance_specular(0.0f);

    for (int i = 0; i < n_specular_samples; ++i)
    {
        const glm::vec2 xi = Hammersley(i, n_specular_samples);
        const glm::vec3 reflect_dir = glm::normalize(glm::reflect(ray.direction, normal));
        const glm::vec3 sample_dir = importance_sample_GGX(xi, reflect_dir, hit->material().roughness);

        const float NdotL = glm::dot(normal, sample_dir);
        if (NdotL > 0.0f)
        {
            rt::Ray sample_ray;
            sample_ray.origin = intersection;
            sample_ray.direction = sample_dir;

            RayPayload payload;
            this->trace_ray(sample_ray, recursion - 1, t_max, &payload);
            ++this->traced_rays[omp_get_thread_num()];

            irradiance_specular += payload.is_light ? glm::vec3(0.0f) : payload.color * NdotL;
            total_weight += NdotL;
        }
    }

    // integrate BRDF belongs to indirect specular
    const float NdotV = glm::max(glm::dot(normal, view), 0.0f);
    glm::vec3 v(sqrt(1.0f - NdotV * NdotV), 0.0f, NdotV);
    glm::vec3 n(0.0f, 0.0f, 1.0f);
    glm::vec2 BRDF(0.0f);

    for (int i = 0; i < 256; ++i)
    {
        glm::vec2 xi = Hammersley(i, 256);
        glm::vec3 h = importance_sample_GGX(xi, n, hit->material().roughness);
        glm::vec3 l = glm::normalize(2.0f * glm::dot(v, h) * h - v);

        const float NdotL = glm::max(l.z, 0.0f);
        const float NdotH = glm::max(h.z, 0.0f);
        const float VdotH = glm::max(glm::dot(v, h), 0.0f);

        if (NdotL > 0.0f)
        {
            const float G = geometry_GGX(n, v, l, hit->material().roughness);
            const float G_vis = (G * VdotH) / (NdotH * NdotV);
            const float fc = pow(1.0f - VdotH, 5.0f);

            BRDF.x += (1.0f - fc) * G_vis;
            BRDF.y += fc * G_vis;
        }
    }
    irradiance_specular /= total_weight;
    BRDF /= 256.0f;

    glm::vec3 F0(0.04f);
    F0 = glm::mix(F0, hit_sphere->material().albedo, hit_sphere->material().metallic);

    glm::vec3 ks = fresnel_schlick_roughness(glm::max(glm::dot(normal, view), 0.0f), F0, hit_sphere->material().roughness);
    glm::vec3 kd = 1.0f - ks;
    kd *= 1.0f - hit_sphere->material().metallic;

    glm::vec3 diffuse = kd * irradiance_diffuse;
    glm::vec3 specular = irradiance_specular * (ks * BRDF.x + BRDF.y);

    ((RayPayload*)ray_payload)->color = diffuse;
    std::this_thread::yield();
}

void RayTracerV2::miss_shader(const rt::Ray& ray, int recursuon, float t_max, void* ray_payload)
{
    constexpr glm::vec2 inv_atan(0.1591, 0.3183);
    glm::vec2 uv(atan2(ray.direction.z, ray.direction.x), asin(ray.direction.y));
    uv = uv * inv_atan + 0.5f;
    uv.y = 1.0f - uv.y;

    const glm::ivec2 px_uv(uv.x * this->env_width, uv.y * this->env_height);
    ((RayPayload*)ray_payload)->color = this->environment[px_uv.y * this->env_width + px_uv.x];
    //((RayPayload*)ray_payload)->color = { 0.0f, 0.0f, 0.0f };
    ((RayPayload*)ray_payload)->is_light = false;
}


void RayTracerV2::run_app(void)
{
    using namespace std::chrono;

    std::atomic_bool should_print{ true };
    std::thread print_thread(RayTracerV2::print_rendered_pixel, &should_print, &this->rendered_pixel, this->traced_rays);

    time_point<high_resolution_clock> t0 = high_resolution_clock::now();
    this->run();
    time_point<high_resolution_clock> t1 = high_resolution_clock::now();
    int64_t t_render_ms = duration_cast<milliseconds>(t1 - t0).count();
    double t_render_s = (double)t_render_ms / 1000.0;
    
    should_print = false;
    print_thread.join();

    uint64_t traced_rays = 0;
    for (uint32_t i = 0; i < n_threads; i++)
    {
        traced_rays += this->traced_rays[i];
    }

    std::cout << "\rRendering done: " << N_PIXELS << "/" << N_PIXELS << " pixel.                    ";
    std::cout << std::endl << "Render time: " << t_render_s << " seconds" << std::endl;
    std::cout << "Traced rays: " << traced_rays << std::endl;
    std::cout << "Ray throughput: " << static_cast<uint32_t>(traced_rays / t_render_s) << " rays / second" << std::endl;
}

glm::vec3 RayTracerV2::compute_direction(float x, float y, float aspect, const glm::vec3& dir, const glm::vec3& up, float fov)
{
    // combute view
    glm::mat3 view;
    view[2] = glm::normalize(dir);
    view[0] = glm::normalize(glm::cross(up, view[2]));
    view[1] = glm::cross(view[2], view[0]);

    // combute perspective
    const float d = aspect / tan(fov / 2);
    glm::vec3 rd = glm::vec3(x * aspect, y, 0.0f) - glm::vec3(0.0f, 0.0f, -d);

    return view * glm::normalize(rd);
}

glm::vec3 RayTracerV2::vogeldisk_sphere(int sample_index, int samples_count, float phi)
{
    float theta = 2.4f * (float)sample_index + phi;
    float r = sqrt((float)sample_index + 0.5f) / sqrt((float)samples_count);
    glm::vec2 u = r * glm::vec2(fast_cos(theta), fast_sin(theta));
    return glm::vec3(u.x, u.y, fast_cos(r * (M_PI / 2)));
}

glm::vec2 RayTracerV2::vogeldisk(int sample_index, int samples_count, float phi)
{
    float theta = 2.4f * (float)sample_index + phi;
    float r = sqrt((float)sample_index + 0.5f) / sqrt((float)samples_count);
    return r * glm::vec2(fast_cos(theta), fast_sin(theta));
}


glm::vec3 RayTracerV2::light_sample(const glm::vec2& xi, const glm::vec3& l_direction, float l_distance, float l_radius)
{
    const float phi = xi.x * 2.0f * M_PI;
    const float theta = 0.5f * M_PI - sqrt(xi.y) * fast_asin(l_radius / l_distance);     

    const glm::vec3 dir(fast_cos(theta) * fast_cos(phi), fast_cos(theta) * fast_sin(phi), fast_sin(theta));
    glm::vec3 up = abs(l_direction.z) < 0.999 ? glm::vec3(0.0, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0);

    glm::mat3 TBN;
    TBN[2] = glm::normalize(l_direction);
    TBN[0] = glm::normalize(glm::cross(up, TBN[2]));
    TBN[1] = glm::cross(TBN[2], TBN[0]);

    return TBN * dir;
}

glm::vec3 RayTracerV2::importance_sample_GGX(glm::vec2 Xi, glm::vec3 R, float roughness)
{
    float a = roughness * roughness;

    float phi = 2.0 * M_PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // from spherical coordinates to cartesian coordinates
    glm::vec3 H;
    H.x = fast_cos(phi) * sinTheta;
    H.y = fast_sin(phi) * sinTheta;
    H.z = cosTheta;

    // from tangent-space vector to world-space sample vector
    glm::vec3 up = glm::normalize(abs(R.z) < 0.999 ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 tangent = glm::normalize(glm::cross(up, R));
    glm::vec3 bitangent = glm::cross(R, tangent);

    glm::vec3 sampleVec = tangent * H.x + bitangent * H.y + R * H.z;
    return glm::normalize(sampleVec);
}

float RayTracerV2::geometry_sub_GGX(const glm::vec3& n, const glm::vec3& x, float k)
{
    const float NdotX = glm::max(glm::dot(n, x), 0.0f);
    const float denom = NdotX * (1.0f - k) + k;
    return NdotX / denom;
}

float RayTracerV2::geometry_GGX(const glm::vec3& n, const glm::vec3& v, const glm::vec3& l, float a)
{
    float k = (a * a) / 2.0f;
    return geometry_sub_GGX(n, v, k) * geometry_sub_GGX(n, l, k);
}

glm::vec3 RayTracerV2::fresnel_schlick_roughness(float cos_theta, const glm::vec3& F0, float roughness)
{
    return F0 + (glm::max(glm::vec3(1.0f - roughness), F0) - F0) * pow(glm::max(1.0f - cos_theta, 0.0f), 5.0f);
}


inline bool RayTracerV2::is_light(const rt::Primitive* p)
{
    return (p->material().roughness < 0.0f);
}

void RayTracerV2::print_rendered_pixel(std::atomic_bool* should_print, std::atomic_uint32_t* px, uint64_t* traced_rays)
{
    while (*should_print)
    {
        std::cout << "\rRendering done: " << *px << "/" << N_PIXELS << " pixel.            ";
        std::this_thread::yield();
    }
}

void RayTracerV2::load_lookup(void)
{
    for (int i = 0; i < 36000; i++)
    {
        sin_lookup[i] = sin((i * M_PI) / 18000.0f);
        cos_lookup[i] = cos((i * M_PI) / 18000.0f);
    }

    for (int i = 0; i < 9000; i++)
    {
        asin_lookup[i] = asin(i / 9000.0f);
    }
}

inline float RayTracerV2::fast_sin(float x)
{
    return sin_lookup[(size_t)((x * 18000.0f) / M_PI) % 36000];
}

inline float RayTracerV2::fast_cos(float x)
{
    return cos_lookup[(size_t)((x * 18000.0f) / M_PI) % 36000];
}

inline float RayTracerV2::fast_asin(float x)
{
    return asin_lookup[(size_t)(x * 9000)];
}
