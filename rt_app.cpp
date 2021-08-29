#include "rt_app.h"
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <thread>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>

#define SCR_WIDTH       (960 * 1)
#define SCR_HEIGHT      (540 * 1)
#define SCR_ASPECT      ((float)SCR_WIDTH / (float)SCR_HEIGHT)
#define N_PIXELS        (SCR_WIDTH * SCR_HEIGHT)
#define N_PIXELS_100    (N_PIXELS / 100)
#define PRIM_COUNT      3

#define M_PI_2_F    1.5707963f
#define M_2_PI_F    6.2831853f
#define M_PI_F      3.1415926f

float RadicalInverse_VdC(uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10; // / 0x100000000
}

glm::vec2 Hammersley(uint32_t i, uint32_t N)
{
    return glm::vec2((float)i / (float)N, RadicalInverse_VdC(i));
}
// ----------------------------------------------------------------------------

PathTracer::PathTracer(void)
{
    rt::ImageCreateInfo fbo_ci = {};
    fbo_ci.width = SCR_WIDTH;
    fbo_ci.height = SCR_HEIGHT;
    this->set_framebuffer(fbo_ci);
    this->set_num_threads(n_threads);

    rt::BufferLayout buffer_layout = {};
    buffer_layout.size = spheres.size();
    buffer_layout.first = 0;
    buffer_layout.last = buffer_layout.size;

    rt::Buffer primitive_buffer(buffer_layout);
    primitive_buffer.data(0, spheres.size(), spheres.data());

    this->draw_buffer(primitive_buffer);

    this->rendered_pixel = 0;
    this->traced_rays = new uint64_t[this->n_threads];
    memset(this->traced_rays, 0, sizeof(uint64_t) * this->n_threads);
}

PathTracer::~PathTracer(void)
{
     const rt::Framebuffer& fbo = this->get_framebuffer();
     stbi_write_png("../../../output.png", fbo.width(), fbo.height(), 3, fbo.map_rdonly(), fbo.width() * 3 * sizeof(uint8_t));

     delete[] this->traced_rays;
}


void PathTracer::init_static(void)
{
    n_threads = omp_get_max_threads() * 4;
    load_textures();
    load_primitives();
}

void PathTracer::load_textures(void)
{
    environment.set_address_mode(rt::RT_TEXTURE_ADDRESS_MODE_CLAMP_TO_BORDER, rt::RT_TEXTURE_ADDRESS_MODE_CLAMP_TO_BORDER, rt::RT_TEXTURE_ADDRESS_MODE_CLAMP_TO_BORDER);
    environment.set_filter(rt::RT_FILTER_LINEAR);
    brdf_lookup.set_address_mode(rt::RT_TEXTURE_ADDRESS_MODE_CLAMP_TO_EDGE, rt::RT_TEXTURE_ADDRESS_MODE_CLAMP_TO_EDGE, rt::RT_TEXTURE_ADDRESS_MODE_CLAMP_TO_BORDER);
    brdf_lookup.set_filter(rt::RT_FILTER_LINEAR);
    if (rt::TextureLoader::loadf(environment, "../../../assets/skyboxes/environment.hdr", 3) != rt::RT_IMAGE_ERROR_NONE)
        throw std::runtime_error("Failed to load environment.");
}

void PathTracer::load_primitives(void)
{
    std::ifstream objects("../../../assets/objects.txt"), materials("../../../assets/materials.txt");
    if (!objects || !materials)
        throw std::runtime_error("Failed to load primitives.");

    while (!(objects.eof() || materials.eof()))
    {
        glm::vec3 pos;
        float r;
        Material mtl;

        objects >> pos.x >> pos.y >> pos.z >> r;
        materials >> mtl.albedo().r >> mtl.albedo().g >> mtl.albedo().b 
                  >> mtl.emission().r >> mtl.emission().g >> mtl.emission().b 
                  >> mtl.opacity() >> mtl.roughness() >> mtl.metallic();

        spheres.push_back(rt::Sphere(pos, r, mtl));
    }
}


glm::vec3 PathTracer::compute_direction(const glm::vec2& ndc, float aspect, const glm::vec3& dir, const glm::vec3& up, float fov)
{
    // combute view
    glm::mat3 view;
    view[2] = glm::normalize(dir);
    view[0] = glm::normalize(glm::cross(up, view[2]));
    view[1] = glm::cross(view[2], view[0]);

    // combute perspective
    const float d = aspect / tan(fov / 2);
    glm::vec3 rd = glm::vec3(ndc.x * aspect, ndc.y, 0.0f) - glm::vec3(0.0f, 0.0f, -d);

    return view * glm::normalize(rd);
}

PathTracer::SampleType PathTracer::generate_sample_type(const Material* mtl, const glm::vec2& noise_seed)
{
    const float x = noise(noise_seed);

    const float th_a = 1.0f - mtl->opacity();
    const float th_d = (0.5f - (mtl->metallic() / 2.0f)) * mtl->opacity();

    if (x < th_a)                           return SAMPLE_TYPE_TRANSPARENCY;
    if (x >= th_a && x < (th_a + th_d))     return SAMPLE_TYPE_DIFFUSE;
    return SAMPLE_TYPE_SPECULAR;
}

void PathTracer::print_rendered_pixel(std::atomic_bool* should_print, std::atomic_uint32_t* px, uint64_t* traced_rays)
{
    while (*should_print)
    {
        std::cout << "\rRendering done: " << *px << "/" << N_PIXELS << " pixel.            ";
        std::this_thread::yield();
    }
}


glm::vec3 PathTracer::light_sample(const glm::vec2& xi, const glm::vec3& l_direction, float l_distance, float l_radius)
{
    const float phi = xi.x * 2.0f * M_PI_F;
    const float theta = M_PI_2_F - sqrt(xi.y) * asin(l_radius / l_distance);     

    const glm::vec3 dir(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta));
    glm::vec3 up = abs(l_direction.z) < 0.999 ? glm::vec3(0.0, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0);

    glm::mat3 TBN;
    TBN[2] = glm::normalize(l_direction);
    TBN[0] = glm::normalize(glm::cross(up, TBN[2]));
    TBN[1] = glm::cross(TBN[2], TBN[0]);

    return TBN * dir;
}

glm::vec3 PathTracer::importance_sample_GGX(glm::vec2 Xi, glm::vec3 n, float roughness)
{
    float a = roughness * roughness;

    float phi = 2.0 * M_PI_F * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // from spherical coordinates to cartesian coordinates
    glm::vec3 h;
    h.x = cos(phi) * sinTheta;
    h.y = sin(phi) * sinTheta;
    h.z = cosTheta;

    // from tangent-space vector to world-space sample vector
    glm::vec3 up = glm::normalize(abs(n.z) < 0.999 ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat3 TBN;
    TBN[2] = glm::normalize(n);
    TBN[0] = glm::normalize(glm::cross(up, TBN[2]));
    TBN[1] = glm::cross(TBN[2], TBN[0]);

    glm::vec3 sampleVec = TBN * h;
    return glm::normalize(sampleVec);
}

inline float PathTracer::geometry_sub_GGX(const glm::vec3& n, const glm::vec3& x, float k)
{
    const float NdotX = glm::max(glm::dot(n, x), 0.0f);
    const float denom = NdotX * (1.0f - k) + k;
    return NdotX / denom;
}

inline float PathTracer::geometry_GGX(const glm::vec3& n, const glm::vec3& v, const glm::vec3& l, float roughness)
{
    float k = (roughness * roughness) / 2.0f;
    return geometry_sub_GGX(n, v, k) * geometry_sub_GGX(n, l, k);
}

inline glm::vec3 PathTracer::fresnel_schlick(float cos_theta, const glm::vec3& F0)
{
    return F0 + (1.0f - F0) * powf(1.0f - cos_theta, 5.0f);
}

inline glm::vec3 PathTracer::fresnel_schlick_inverted(float cos_theta, const glm::vec3& F0, const Material& mtl)
{
    glm::vec3 F_inverted = 1.0f - fresnel_schlick(cos_theta, F0);
    return F_inverted *= (1.0f - mtl.metallic());
}

inline float PathTracer::noise(glm::vec2 pos)
{
    return glm::fract(sin(glm::dot(pos, glm::vec2(12.9898, 78.233))) * 43758.5453);
}


inline uint32_t PathTracer::float2bits(float f)
{
    uint32_t i;
    memcpy(&i, &f, sizeof(float));
    return i;
}

inline float PathTracer::bits2float(uint32_t i)
{
    float f;
    memcpy(&f, &i, sizeof(uint32_t));
    return f;
}

inline float PathTracer::next_float_up(float f)
{
    // 2 special cases:
    // if x is positive infinity, then the value can't be incremented any further and it is returned unchanged
    // if x is negative 0, change it to positive 0, as negative 0 has a different memory representation
    if (std::isinf(f) && f > 0.0f)
        return f;
    if (f == -0.0f)
        f = 0.0f;

    // advance x to the next higher float
    // this can only be done because of the memory representation of the IEEE754 standard
    uint32_t i = float2bits(f);
    if (f >= 0) ++i;
    else        --i;
    return bits2float(i);
}

inline float PathTracer::next_float_down(float f)
{
    // is the same as next_float_up but in reverse
    if (std::isinf(f) && f < 0.0f)
        return f;
    if (f == 0.0f)
        f = -0.0f;

    uint32_t i = float2bits(f);
    if (f <= 0) ++i;
    else        --i;
    return bits2float(i);
}

inline glm::vec3 PathTracer::offset_ray_origin(const glm::vec3& p, const glm::vec3& n, const glm::vec3& w)
{
    float d = 2e-4f;
    glm::vec3 offset = d * n;
    if (dot(w, n) < 0)
        offset = -offset;
    glm::vec3 po = p + offset;

    if (offset.x > 0)       po.x = next_float_up(po.x);
    else if (offset.x < 0)  po.x = next_float_down(po.x);
    if (offset.y > 0)       po.y = next_float_up(po.y);
    else if (offset.y < 0)  po.y = next_float_down(po.y);
    if (offset.z > 0)       po.z = next_float_up(po.z);
    else if (offset.z < 0)  po.z = next_float_down(po.z);

    return po;
}


glm::vec3 PathTracer::ray_generation_shader(uint32_t x, uint32_t y)
{
    float ndc_x = gl::convert::from_pixels_pos_x(x, SCR_WIDTH) + (0.5f / SCR_WIDTH);
    float ndc_y = gl::convert::from_pixels_pos_y(y, SCR_HEIGHT) - (0.5f / SCR_HEIGHT);

    const glm::vec3 cam_pos = glm::vec3(0.0f, 0.0f, 10.0f);
    const glm::vec3 look_at(0.0f, 0.0f, 0.0f);
    const glm::vec3 up(0.0f, 1.0f, 0.0f);

    rt::ray_t ray;
    ray.origin = cam_pos;

    RayPayload payload;
    payload.ray_type = RAY_TYPE_PRIMARY;
    payload.noise_coord = glm::vec2(x * y, 0.0f);

    glm::vec3 irradiance_diffuse(0.0f);
    glm::vec3 irradiance_specular(0.0f);
    glm::vec3 direct_light(0.0f);
    glm::vec3 emission(0.0f);

    bool miss = false;
    for (int i = 0; i < SAMPLE_COUNT && !miss; i++)
    {
        glm::vec2 offset = (i == 0) ? glm::vec2(0.0f) : (Hammersley(i, SAMPLE_COUNT) - 0.5f) / glm::vec2(SCR_WIDTH, SCR_HEIGHT);
        ray.direction = this->compute_direction(glm::vec2(ndc_x, ndc_y)  + offset, SCR_ASPECT, look_at - cam_pos, up, glm::radians(100.0f));

        payload.noise_coord.y = i * (7 * ITERATIONS);
        this->trace_ray(ray, ITERATIONS, T_MAX, rt::RT_CULL_MASK_NONE, &payload);
        ++this->traced_rays[omp_get_thread_num()];
        miss = (i == 0 && payload.t >= T_MAX);

        emission            += payload.emission;            // emissive properties of the shading point
        direct_light        += payload.direct_light;        // light that is directly hitting the shading point from all directions
        irradiance_diffuse  += payload.indirect_light;      // indirect light hitting the shading point from all directions
        irradiance_specular += payload.specular;            // reflections of the shading point
    }
    if (!miss)
    {
        emission            /= SAMPLE_COUNT;
        irradiance_diffuse  /= SAMPLE_COUNT;
        direct_light        /= SAMPLE_COUNT;
        irradiance_specular /= SAMPLE_COUNT;
    }
    ++this->rendered_pixel;

    glm::vec3 irradiance = emission + direct_light + irradiance_diffuse + irradiance_specular;
    glm::vec3 LDR = irradiance / (irradiance + 1.0f);
    return glm::pow(LDR, glm::vec3(1.0f / 2.2f));
}

void PathTracer::closest_hit_shader(const rt::ray_t& ray, int recursion, float t, float t_max, const rt::Primitive* hit, rt::RayHitInformation hit_info, void* ray_payload)
{
    rt::Sphere* hit_sphere = (rt::Sphere*)hit;
    const Material* mtl = dynamic_cast<const Material*>(hit->attribute());
    RayPayload* payload_in = (RayPayload*)ray_payload;

    payload_in->t = t;
    --recursion;

    const glm::vec3 intersection = ray.origin + t * ray.direction;  // prevent self-intersection
    const glm::vec3 normal = glm::normalize(intersection - hit_sphere->center());
    const glm::vec3 view = -ray.direction;

    // emissive properties of the current shading point
    payload_in->emission = mtl->emission();

    // If the last iteration hits something, only the emissive properties are returned,
    // there are no further calculations needed.
    // The same counts for tracing shadow rays. They are used to sample the light source and their
    // tracing process ends, if they hit something, no matter how many iterations there are left.
    if (recursion == 0 || payload_in->ray_type == RAY_TYPE_SHADOW) return;

    // matrix to transform a vector from a local space to the world space
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::mat3 TBN;
    TBN[2] = glm::normalize(normal);
    TBN[0] = glm::normalize(glm::cross(up, TBN[2]));
    TBN[1] = cross(TBN[2], TBN[0]);

    // ray object used to trace the next ray(s)
    rt::ray_t sample_ray;

    // payload: values returned by the next iteration
    RayPayload payload;
    payload.noise_coord = payload_in->noise_coord;

    // base reflectivity
    glm::vec3 F0(0.04f);
    F0 = glm::mix(F0, mtl->albedo(), mtl->metallic());

    // outgoing direct lighting
    {
        const glm::vec3 light_pos(0.0f, 4.0f, 0.0f);
        const glm::vec3 light_direction = glm::normalize(light_pos - intersection);
        const float light_radius = 1.0f;
        const float light_distance = glm::distance(intersection, light_pos);

        // noise coordinates for shadow ray tracing
        const glm::vec2 s1(0.0f, (recursion - 1) * 7.0f + 0.0f);
        const glm::vec2 s2(0.0f, (recursion - 1) * 7.0f + 1.0f);

        payload.ray_type = RAY_TYPE_SHADOW;
        const glm::vec2 xi(noise(payload_in->noise_coord + s1), noise(payload_in->noise_coord + s2));
        sample_ray.direction = light_sample(xi, light_direction, light_distance, light_radius);
        sample_ray.origin = offset_ray_origin(intersection, normal, sample_ray.direction);
        this->trace_ray(sample_ray, recursion, t_max, rt::RT_CULL_MASK_NONE, &payload);
        ++this->traced_rays[omp_get_thread_num()];

        const glm::vec3 h = glm::normalize(view + sample_ray.direction);
        const float VdotH = glm::clamp(glm::dot(view, h), 0.0f, 1.0f);

        const glm::vec3 kd = fresnel_schlick_inverted(VdotH, F0, *mtl);
        const glm::vec3 brdf = kd * (mtl->albedo() / M_PI_F);

        // Only the incoming emission part is used as all other parts as direct light, indirect light and specular light will not contribute.
        payload.emission *= (1.0f / (1.0f + payload.t));
        payload_in->direct_light = payload.emission * glm::max(glm::dot(normal, sample_ray.direction), 0.0f) * brdf;
    }

    // The last iteration will always go towards the lightsource(s).
    // In other words, the last ray is always a shadow ray, so there is no
    // need to combute indirect lighting or specular lighting.
    if (recursion == 1) return;

    payload = {};
    payload.noise_coord = payload_in->noise_coord;

    const glm::vec2 s3(0.0f, (recursion-1) * 7.0f + 2.0f);
    payload.ray_type = RAY_TYPE_SECONDARY;
    SampleType sample_type = generate_sample_type(mtl, payload_in->noise_coord + s3);

    // outgoing indirect lighting
    if (sample_type == SAMPLE_TYPE_DIFFUSE)
    {
        // generate cosine-distributed sample on a hemisphere
        const glm::vec2 s4(0.0f, (recursion-1) * 7.0f + 3.0f);
        const glm::vec2 s5(0.0f, (recursion-1) * 7.0f + 4.0f);
        const float theta = acos(sqrt(noise(payload_in->noise_coord + s4)));
        const float phi = M_2_PI_F * noise(payload_in->noise_coord + s5);
        glm::vec3 v(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)); // cosine weighted distribution

        // orientate hemisphere around the normal vector and trace the ray
        sample_ray.direction = TBN * v;
        sample_ray.origin = offset_ray_origin(intersection, normal, sample_ray.direction);
        this->trace_ray(sample_ray, recursion, t_max, rt::RT_CULL_MASK_NONE , &payload);
        ++this->traced_rays[omp_get_thread_num()];

        const glm::vec3 h = glm::normalize(view + sample_ray.direction);
        const float VdotH = glm::clamp(glm::dot(view, h), 0.0f, 1.0f);
        
        const glm::vec3 kd = fresnel_schlick_inverted(VdotH, F0, *mtl);
        const glm::vec3 brdf = kd * mtl->albedo();

        // indirect light ray may hit the light source directly then the attenutation must be calculated
        payload.emission *= (1.0f / (1.0f + payload.t));
        const glm::vec3 incoming_radiance = payload.emission + payload.direct_light + payload.indirect_light + payload.specular;
        payload_in->indirect_light = incoming_radiance * brdf;
    }
    // outgoing specular lighting (reflection)
    else if (sample_type == SAMPLE_TYPE_SPECULAR)
    {
        const glm::vec2 s6(0.0f, (recursion-1) * 7.0f + 5.0f);
        const glm::vec2 s7(0.0f, (recursion-1) * 7.0f + 6.0f);
        const glm::vec2 xi(noise(payload_in->noise_coord + s6), noise(payload_in->noise_coord + s7));
        const glm::vec3 h = importance_sample_GGX(xi, normal, mtl->roughness());
        sample_ray.direction = glm::reflect(ray.direction, h);
        sample_ray.origin = offset_ray_origin(intersection, normal, sample_ray.direction);

        const float NdotL = glm::max(glm::dot(normal, sample_ray.direction), 0.0f);
        if (NdotL > 0.0f)
        {
            this->trace_ray(sample_ray, recursion, t_max, rt::RT_CULL_MASK_NONE, &payload);
            ++this->traced_rays[omp_get_thread_num()];

            // calculate BRDF
            const float NdotV = glm::clamp(glm::dot(normal, view), 0.0f, 1.0f);
            const float NdotH = glm::clamp(glm::dot(normal, h), 0.0f, 1.0f);
            const float VdotH = glm::clamp(glm::dot(view, h), 0.0f, 1.0f);

            const glm::vec3 F = fresnel_schlick(VdotH, F0);
            const float G = geometry_GGX(normal, view, sample_ray.direction, mtl->roughness());
            const glm::vec3 brdf = F * G * VdotH / glm::max((NdotH * NdotV), 0.001f);

            const glm::vec3 incoming_radiance = payload.emission + payload.direct_light + payload.indirect_light + payload.specular;
            payload_in->specular = incoming_radiance * brdf;
        }
    }
}

void PathTracer::miss_shader(const rt::ray_t& ray, int recursuon, float t_max, void* ray_payload)
{
    RayPayload* payload = (RayPayload*)ray_payload;
    const glm::vec3 rd = glm::normalize(ray.direction);
    const glm::vec4 c = environment.sample(glm::vec4(rd.x, rd.y, rd.z, 0.0f));
    payload->indirect_light = c;
    payload->t = t_max;
}

void PathTracer::run_app(void)
{
    using namespace std::chrono;

    std::atomic_bool should_print{ true };
    std::thread print_thread(PathTracer::print_rendered_pixel, &should_print, &this->rendered_pixel, this->traced_rays);

    time_point<high_resolution_clock> t0 = high_resolution_clock::now();
    this->run();
    time_point<high_resolution_clock> t1 = high_resolution_clock::now();
    int64_t t_render_ms = duration_cast<milliseconds>(t1 - t0).count();
    double t_render_s = (double)t_render_ms / 1000.0;

    should_print = false;
    print_thread.join();

    uint64_t traced_rays = 0;
    for (uint32_t i = 0; i < n_threads; i++)
        traced_rays += this->traced_rays[i];

    std::cout << "\rRendering done: " << N_PIXELS << "/" << N_PIXELS << " pixel.                    ";
    std::cout << std::endl << "Render time: " << t_render_s << " seconds" << std::endl;
    std::cout << "Traced rays: " << traced_rays << std::endl;
    std::cout << "Ray throughput: " << static_cast<uint32_t>(traced_rays / t_render_s) << " rays / second" << std::endl;
}
