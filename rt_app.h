#pragma once

#include <rt/ray_tracing.h>
#include "glc-1.0.0/GL/glc.h"
#include <stb/stb_image_write.h>
#include <random>
#include <atomic>

class Material : public rt::PrimitiveAttribute
{
private:
    glm::vec3 _albedo;
    glm::vec3 _emission;
    float _roughness;
    float _metallic;
    float _opacity;

public:
    explicit Material(const glm::vec3& albedo = glm::vec3(0.0f), const glm::vec3& emission = glm::vec3(0.0f), float roughness = 0.0f, float metallic = 0.0f, float opacity = 1.0f)
    {
        this->_albedo = albedo;
        this->_emission = emission;
        this->_roughness = roughness;
        this->_metallic = metallic;
        this->_opacity = opacity;
    }

    Material(const Material& mtl)
    {
        *this = mtl;
    }
    ~Material(void) {}

    Material& operator= (const Material& mtl)
    {
        this->_albedo = mtl._albedo;
        this->_emission = mtl._emission;
        this->_roughness = mtl._roughness;
        this->_metallic = mtl._metallic;
        this->_opacity = mtl._opacity;
        return *this;
    }
    virtual rt::PrimitiveAttribute* clone_dynamic(void) const { return new Material(*this); }

    glm::vec3&          albedo(void)    noexcept        { return this->_albedo; }
    const glm::vec3&    albedo(void)    const noexcept  { return this->_albedo; }
    glm::vec3&          emission(void)  noexcept        { return this->_emission; }
    const glm::vec3&    emission(void)  const noexcept  { return this->_emission; }
    float&              roughness(void) noexcept        { return this->_roughness; }
    float               roughness(void) const noexcept  { return this->_roughness; }
    float&              metallic(void)  noexcept        { return this->_metallic; }
    float               metallic(void)  const noexcept  { return this->_metallic; }
    float&              opacity(void)   noexcept        { return this->_opacity; }
    float               opacity(void)   const noexcept  { return this->_opacity; }
};

class PathTracer : private rt::RayTracer
{
private:
    enum RayType
    {
        RAY_TYPE_PRIMARY,
        RAY_TYPE_SECONDARY,
        RAY_TYPE_SHADOW
    };

    enum SampleType
    {
        SAMPLE_TYPE_DIFFUSE,
        SAMPLE_TYPE_SPECULAR,
        SAMPLE_TYPE_TRANSPARENCY
    };

    struct RayPayload
    {
        glm::vec3 emission{ 0.0f };
        glm::vec3 direct_light{ 0.0f };
        glm::vec3 indirect_light{ 0.0f };
        glm::vec3 specular{ 0.0f };
        float weight{ 1.0f };
        RayType ray_type;
        float t{ 0.0f };
        glm::vec2 noise_coord{ 0.0f };
    };

    std::atomic_uint32_t rendered_pixel;
    uint64_t* traced_rays;

    static constexpr int SAMPLE_COUNT = 1000;
    static constexpr float T_MAX = 100.0f;
    static constexpr int ITERATIONS = 6;

    static inline uint32_t n_threads = 0;
    static inline rt::SphericalMap<float, float> environment;
    static inline rt::Texture2D<float, float> brdf_lookup;
    static inline std::vector<rt::Sphere> spheres;

    static void load_textures(void);
    static void load_brdf(void);
    static void load_primitives(void);

    static glm::vec3 compute_direction(const glm::vec2& ndc, float aspect, const glm::vec3& dir, const glm::vec3& up, float fov);
    static SampleType generate_sample_type(const Material* mtl, const glm::vec2& noise_seed);
    static void print_rendered_pixel(std::atomic_bool* should_print, std::atomic_uint32_t* px, uint64_t* traced_rays);

    static glm::vec3 light_sample(const glm::vec2& xi, const glm::vec3& l_direction, float l_distance, float l_radius);
    static glm::vec3 importance_sample_GGX(glm::vec2 Xi, glm::vec3 N, float roughness);
    static float geometry_sub_GGX(const glm::vec3& n, const glm::vec3& x, float k);
    static float geometry_GGX(const glm::vec3& n, const glm::vec3& v, const glm::vec3& l, float a);
    static glm::vec3 fresnel_schlick_roughness(float cos_theta, const glm::vec3& F0, float roughness);
    static float noise(glm::vec2 pos);

    glm::vec3 ray_generation_shader(uint32_t x, uint32_t y);
    void closest_hit_shader(const rt::ray_t& ray, int recursion, float t, float t_max, const rt::Primitive* hit, uint32_t hit_info, void* ray_payload);
    void miss_shader(const rt::ray_t& ray, int recursuon, float t_max, void* ray_payload);

public:
    PathTracer(void);
    ~PathTracer(void);

    static void init_static(void);
    void run_app(void);
};