#pragma once

#include <rt/ray_tracing.h>
#include "glc-1.0.0/GL/glc.h"
#include <stb/stb_image_write.h>
#include <random>
#include <atomic>
#include <unordered_map>

class Material : public rt::PrimitiveAttribute
{
private:
    glm::vec3 _albedo;
    glm::vec3 _emission;
    float _roughness;
    float _metallic;
    float _opacity;
    float _ior; // index of refraction

    const rt::Texture2D<uint8_t, float>* _albedo_map;
    const rt::Texture2D<uint8_t, float>* _roughness_map;
    const rt::Texture2D<uint8_t, float>* _metallic_map;
    const rt::Texture2D<uint8_t, float>* _normal_map;

public:
    explicit Material(const glm::vec3& albedo = glm::vec3(0.0f), 
                      const glm::vec3& emission = glm::vec3(0.0f), 
                      float roughness = 0.0f, 
                      float metallic = 0.0f, 
                      float opacity = 1.0f, 
                      float ior = 1.0f,
                      const rt::Texture2D<uint8_t, float>* albedo_map = nullptr,
                      const rt::Texture2D<uint8_t, float>* roughness_map = nullptr,
                      const rt::Texture2D<uint8_t, float>* metallic_map = nullptr,
                      const rt::Texture2D<uint8_t, float>* normal_map = nullptr)
    {
        this->_albedo = albedo;
        this->_emission = emission;
        this->_roughness = roughness;
        this->_metallic = metallic;
        this->_opacity = opacity;
        this->_ior = ior;
        this->_albedo_map = albedo_map;
        this->_roughness_map = roughness_map;
        this->_metallic_map = metallic_map;
        this->_normal_map = normal_map;
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
        this->_ior = mtl._ior;
        this->_albedo_map = mtl._albedo_map;
        this->_roughness_map = mtl._roughness_map;
        this->_metallic_map = mtl._metallic_map;
        this->_normal_map = mtl._normal_map;
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
    float&              ior(void)       noexcept        { return this->_ior; }
    float               ior(void)       const noexcept  { return this->_ior; }

    const rt::Texture2D<uint8_t, float>*&   albedo_map(void)    noexcept        { return this->_albedo_map; }
    const rt::Texture2D<uint8_t, float>*    albedo_map(void)    const noexcept  { return this->_albedo_map; }
    const rt::Texture2D<uint8_t, float>*&   roughness_map(void) noexcept        { return this->_roughness_map; }
    const rt::Texture2D<uint8_t, float>*    roughness_map(void) const noexcept  { return this->_roughness_map; }
    const rt::Texture2D<uint8_t, float>*&   metallic_map(void)  noexcept        { return this->_metallic_map; }
    const rt::Texture2D<uint8_t, float>*    metallic_map(void)  const noexcept  { return this->_metallic_map; }
    const rt::Texture2D<uint8_t, float>*&   normal_map(void)    noexcept        { return this->_normal_map; }
    const rt::Texture2D<uint8_t, float>*    normal_map(void)    const noexcept  { return this->_normal_map; }
};

class PathTracer : private rt::RayTracer
{
private:
    // ======================= ENUMS =======================
    enum RayType
    {
        RAY_TYPE_PRIMARY,
        RAY_TYPE_SECONDARY,
        RAY_TYPE_SHADOW
    };

    // ======================= STRUCTS =======================
    struct RayPayload
    {
        glm::vec3 emission{ 0.0f };
        glm::vec3 direct_light{ 0.0f };
        glm::vec3 indirect_light{ 0.0f };
        glm::vec3 specular{ 0.0f };
        glm::vec3 transmission{ 0.0f };
        RayType ray_type;
        float t{ 0.0f };
        glm::vec2 noise_coord{ 0.0f };
    };

    // ======================= NON-STATIC MEMBERS =======================
    std::atomic_uint32_t rendered_pixel;
    uint64_t* traced_rays;

    // ======================= CONSTANTS =======================
    static constexpr int SAMPLE_COUNT = 10000;
    static constexpr float T_MAX = 100.0f;
    static constexpr int ITERATIONS = 6;

    // ======================= GLOBAL MEMBERS =======================
    static inline uint32_t n_threads = 0;
    static inline rt::SphericalMap<float, float> environment;
    static inline rt::Texture2D<float, float> brdf_lookup;
    static inline std::vector<rt::Sphere> spheres;
    static inline std::unordered_map<std::string, rt::Texture2D<uint8_t, float>> albedo_maps;
    static inline std::unordered_map<std::string, rt::Texture2D<uint8_t, float>> roughness_maps;
    static inline std::unordered_map<std::string, rt::Texture2D<uint8_t, float>> metallic_maps;
    static inline std::unordered_map<std::string, rt::Texture2D<uint8_t, float>> normal_maps;

    // ======================= LOAD METHODS =======================
    static void load_textures(void);
    static void load_primitives(void);

    // ======================= UTILITY METHODS =======================
    static glm::vec3 compute_direction(const glm::vec2& ndc, float aspect, const glm::vec3& dir, const glm::vec3& up, float fov);
    static inline glm::vec4 normal_to_uv(const glm::vec3& n);
    static void print_rendered_pixel(std::atomic_bool* should_print, std::atomic_uint32_t* px, uint64_t* traced_rays);

    static glm::vec3 light_sample(const glm::vec2& xi, const glm::vec3& l_direction, float l_distance, float l_radius);
    static glm::vec3 importance_sample_GGX(glm::vec2 Xi, glm::vec3 N, float roughness);
    static inline float geometry_sub_GGX(float cos_theta, float k);
    static inline float geometry_GGX(float NdotV, float NdotL, float roughness);
    static inline glm::vec3 fresnel_schlick(float cos_theta, const glm::vec3& F0);
    static inline glm::vec3 fresnel_schlick_inverted(float cos_theta, const glm::vec3& F0, float metallic);
    static inline float noise(glm::vec2 pos);

    static inline uint32_t float2bits(float f);
    static inline float bits2float(uint32_t i);
    static inline float next_float_up(float f);
    static inline float next_float_down(float f);
    static inline glm::vec3 offset_ray_origin(const glm::vec3& p, const glm::vec3& n, const glm::vec3& w);

    // ======================= SHADERS =======================
    glm::vec3 ray_generation_shader(uint32_t x, uint32_t y);
    void closest_hit_shader(const rt::ray_t& ray, int recursion, float t, float t_max, const rt::Primitive* hit, rt::RayHitInformation hit_info, void* ray_payload);
    void miss_shader(const rt::ray_t& ray, int recursuon, float t_max, void* ray_payload);

public:
    PathTracer(void);
    ~PathTracer(void);

    static void init_static(void);
    void run_app(void);
};