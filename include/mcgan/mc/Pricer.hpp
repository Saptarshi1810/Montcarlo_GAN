#pragma once

#include "../core/Types.hpp"
#include "PathSampler.hpp"
#include <memory>
#include <vector>
#include <functional>

namespace mcgan {
namespace mc {

/**
 * Result of a Monte Carlo pricing computation.
 */
struct PricingResult {
    float price;              // Estimated price/value
    float variance;           // Variance of the estimate
    float standardError;      // Standard error of the estimate
    int numSamples;           // Number of samples used
    float minValue;           // Minimum sample value
    float maxValue;           // Maximum sample value
    double computeTime;       // Time taken in seconds
    
    PricingResult() 
        : price(0.0f)
        , variance(0.0f)
        , standardError(0.0f)
        , numSamples(0)
        , minValue(std::numeric_limits<float>::max())
        , maxValue(std::numeric_limits<float>::lowest())
        , computeTime(0.0)
    {}
    
    float confidenceInterval95() const {
        return 1.96f * standardError;
    }
};

/**
 * Path tracing result for rendering applications.
 */
struct PathResult {
    Vec3 radiance;           // Accumulated radiance along the path
    int bounces;             // Number of bounces in the path
    bool terminated;         // Whether the path was terminated
    float weight;            // Total path weight
    
    PathResult() 
        : radiance(0, 0, 0)
        , bounces(0)
        , terminated(false)
        , weight(1.0f)
    {}
};

/**
 * Scene description for path tracing.
 */
struct Scene {
    Vec3 lightPosition;
    Vec3 lightColor;
    float lightIntensity;
    
    // Simple sphere geometry for basic testing
    struct Sphere {
        Vec3 center;
        float radius;
        Vec3 albedo;
        Vec3 emission;
    };
    
    std::vector<Sphere> spheres;
    
    Scene() 
        : lightPosition(0, 10, 0)
        , lightColor(1, 1, 1)
        , lightIntensity(1.0f)
    {}
};

/**
 * Ray structure for intersection testing.
 */
struct Ray {
    Vec3 origin;
    Vec3 direction;
    
    Ray(const Vec3& o, const Vec3& d) 
        : origin(o), direction(d.normalized()) 
    {}
    
    Vec3 at(float t) const {
        return origin + direction * t;
    }
};

/**
 * Intersection information.
 */
struct Intersection {
    bool hit;
    float t;
    Vec3 point;
    Vec3 normal;
    Vec3 albedo;
    Vec3 emission;
    int objectIndex;
    
    Intersection() 
        : hit(false)
        , t(std::numeric_limits<float>::max())
        , point(0, 0, 0)
        , normal(0, 1, 0)
        , albedo(0.8f, 0.8f, 0.8f)
        , emission(0, 0, 0)
        , objectIndex(-1)
    {}
};

/**
 * Pricer class for Monte Carlo estimation.
 * Supports both financial option pricing and rendering/path tracing applications.
 */
class Pricer {
public:
    Pricer(std::shared_ptr<PathSampler> sampler);
    virtual ~Pricer() = default;
    
    // === Path Tracing Methods ===
    
    /**
     * Estimate radiance using path tracing.
     */
    Vec3 estimateRadiance(const Ray& ray, const Scene& scene, int maxBounces = 5);
    
    /**
     * Trace a single path through the scene.
     */
    PathResult tracePath(const Ray& ray, const Scene& scene, int maxBounces = 5);
    
    /**
     * Render an image using Monte Carlo path tracing.
     */
    std::vector<Vec3> render(const Scene& scene, int width, int height, 
                            int samplesPerPixel, int maxBounces = 5);
    
    // === Option Pricing Methods ===
    
    /**
     * Price a European call option using Monte Carlo.
     */
    PricingResult priceEuropeanCall(float S0, float K, float r, float sigma, 
                                    float T, int numSamples);
    
    /**
     * Price a European put option using Monte Carlo.
     */
    PricingResult priceEuropeanPut(float S0, float K, float r, float sigma, 
                                   float T, int numSamples);
    
    /**
     * Price an Asian option (average price).
     */
    PricingResult priceAsianOption(float S0, float K, float r, float sigma, 
                                   float T, int numSteps, int numSamples, bool isCall = true);
    
    /**
     * Price a path-dependent option using custom payoff function.
     */
    PricingResult pricePathDependent(
        float S0, float r, float sigma, float T, int numSteps, int numSamples,
        std::function<float(const std::vector<float>&)> payoffFunc);
    
    // === General Monte Carlo Methods ===
    
    /**
     * Estimate an integral using Monte Carlo.
     */
    PricingResult estimateIntegral(
        std::function<float(const std::vector<float>&)> integrand,
        int dimension, int numSamples);
    
    /**
     * Estimate expectation E[f(X)] where X ~ distribution.
     */
    PricingResult estimateExpectation(
        std::function<float(float)> func,
        int numSamples);
    
    // === Configuration ===
    
    void setSampler(std::shared_ptr<PathSampler> sampler) { m_sampler = sampler; }
    void setSeed(int seed) { m_sampler->reset(seed); }
    void setVerbose(bool verbose) { m_verbose = verbose; }
    
    // === Statistics ===
    
    int getTotalSamples() const { return m_totalSamples; }
    int getTotalPaths() const { return m_totalPaths; }
    void resetStatistics() { m_totalSamples = 0; m_totalPaths = 0; }

protected:
    std::shared_ptr<PathSampler> m_sampler;
    bool m_verbose;
    int m_totalSamples;
    int m_totalPaths;
    
    // === Helper Methods ===
    
    // Scene intersection
    Intersection intersect(const Ray& ray, const Scene& scene) const;
    bool intersectSphere(const Ray& ray, const Scene::Sphere& sphere, float& t) const;
    
    // Direct lighting calculation
    Vec3 computeDirectLighting(const Intersection& hit, const Scene& scene);
    
    // Stock price simulation
    float simulateStockPrice(float S0, float r, float sigma, float T);
    std::vector<float> simulateStockPath(float S0, float r, float sigma, float T, int numSteps);
    
    // Statistical utilities
    void computeStatistics(const std::vector<float>& samples, PricingResult& result);
    float mean(const std::vector<float>& samples) const;
    float variance(const std::vector<float>& samples, float mean) const;
};

// === Implementation ===

inline Pricer::Pricer(std::shared_ptr<PathSampler> sampler)
    : m_sampler(sampler)
    , m_verbose(false)
    , m_totalSamples(0)
    , m_totalPaths(0)
{
}

inline Vec3 Pricer::estimateRadiance(const Ray& ray, const Scene& scene, int maxBounces) {
    PathResult result = tracePath(ray, scene, maxBounces);
    return result.radiance;
}

inline PathResult Pricer::tracePath(const Ray& ray, const Scene& scene, int maxBounces) {
    PathResult result;
    
    Ray currentRay = ray;
    Vec3 throughput(1, 1, 1);
    
    for (int bounce = 0; bounce < maxBounces; bounce++) {
        result.bounces = bounce;
        
        // Intersect scene
        Intersection hit = intersect(currentRay, scene);
        
        if (!hit.hit) {
            // Hit environment/sky
            result.terminated = true;
            break;
        }
        
        // Add emission
        result.radiance = result.radiance + throughput * hit.emission;
        
        // Russian Roulette for path termination
        float rrProbability = std::min(1.0f, std::max(throughput.x(), std::max(throughput.y(), throughput.z())));
        if (bounce > 3 && !m_sampler->russianRoulette(rrProbability)) {
            result.terminated = true;
            break;
        }
        
        if (bounce > 3) {
            throughput = throughput / rrProbability;
        }
        
        // Sample next direction
        Vec3 nextDir = m_sampler->sampleHemisphere(hit.normal);
        float pdf = m_sampler->pdf(nextDir, hit.normal);
        
        if (pdf <= 0.0f) {
            result.terminated = true;
            break;
        }
        
        // Update throughput with BRDF and cosine term
        float cosTheta = std::max(0.0f, nextDir.dot(hit.normal));
        throughput = throughput * hit.albedo * (cosTheta / pdf);
        
        // Create next ray
        currentRay = Ray(hit.point + hit.normal * 0.001f, nextDir);
    }
    
    result.weight = throughput.length();
    m_totalPaths++;
    
    return result;
}

inline std::vector<Vec3> Pricer::render(const Scene& scene, int width, int height, 
                                       int samplesPerPixel, int maxBounces) {
    std::vector<Vec3> image(width * height, Vec3(0, 0, 0));
    
    float aspectRatio = float(width) / float(height);
    float fov = 60.0f * M_PI / 180.0f;
    float scale = std::tan(fov * 0.5f);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3 color(0, 0, 0);
            
            for (int s = 0; s < samplesPerPixel; s++) {
                // Pixel coordinates with jittering
                float px = (2.0f * (x + m_sampler->sample1D()) / width - 1.0f) * aspectRatio * scale;
                float py = (1.0f - 2.0f * (y + m_sampler->sample1D()) / height) * scale;
                
                // Create camera ray
                Vec3 origin(0, 0, 0);
                Vec3 direction(px, py, -1);
                direction = direction.normalized();
                
                Ray ray(origin, direction);
                color = color + estimateRadiance(ray, scene, maxBounces);
            }
            
            image[y * width + x] = color / float(samplesPerPixel);
        }
    }
    
    return image;
}

inline PricingResult Pricer::priceEuropeanCall(float S0, float K, float r, float sigma, 
                                               float T, int numSamples) {
    std::vector<float> payoffs(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        float ST = simulateStockPrice(S0, r, sigma, T);
        payoffs[i] = std::max(0.0f, ST - K);
    }
    
    PricingResult result;
    computeStatistics(payoffs, result);
    
    // Discount to present value
    result.price *= std::exp(-r * T);
    result.standardError *= std::exp(-r * T);
    
    m_totalSamples += numSamples;
    return result;
}

inline PricingResult Pricer::priceEuropeanPut(float S0, float K, float r, float sigma, 
                                              float T, int numSamples) {
    std::vector<float> payoffs(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        float ST = simulateStockPrice(S0, r, sigma, T);
        payoffs[i] = std::max(0.0f, K - ST);
    }
    
    PricingResult result;
    computeStatistics(payoffs, result);
    
    result.price *= std::exp(-r * T);
    result.standardError *= std::exp(-r * T);
    
    m_totalSamples += numSamples;
    return result;
}

inline PricingResult Pricer::priceAsianOption(float S0, float K, float r, float sigma, 
                                              float T, int numSteps, int numSamples, bool isCall) {
    std::vector<float> payoffs(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        std::vector<float> path = simulateStockPath(S0, r, sigma, T, numSteps);
        float average = mean(path);
        
        if (isCall) {
            payoffs[i] = std::max(0.0f, average - K);
        } else {
            payoffs[i] = std::max(0.0f, K - average);
        }
    }
    
    PricingResult result;
    computeStatistics(payoffs, result);
    
    result.price *= std::exp(-r * T);
    result.standardError *= std::exp(-r * T);
    
    m_totalSamples += numSamples;
    return result;
}

inline PricingResult Pricer::pricePathDependent(
    float S0, float r, float sigma, float T, int numSteps, int numSamples,
    std::function<float(const std::vector<float>&)> payoffFunc) {
    
    std::vector<float> payoffs(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        std::vector<float> path = simulateStockPath(S0, r, sigma, T, numSteps);
        payoffs[i] = payoffFunc(path);
    }
    
    PricingResult result;
    computeStatistics(payoffs, result);
    
    result.price *= std::exp(-r * T);
    result.standardError *= std::exp(-r * T);
    
    m_totalSamples += numSamples;
    return result;
}

inline PricingResult Pricer::estimateIntegral(
    std::function<float(const std::vector<float>&)> integrand,
    int dimension, int numSamples) {
    
    std::vector<float> values(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        std::vector<float> point(dimension);
        for (int d = 0; d < dimension; d++) {
            point[d] = m_sampler->sample1D();
        }
        values[i] = integrand(point);
    }
    
    PricingResult result;
    computeStatistics(values, result);
    
    m_totalSamples += numSamples;
    return result;
}

inline PricingResult Pricer::estimateExpectation(
    std::function<float(float)> func, int numSamples) {
    
    std::vector<float> values(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        float x = m_sampler->sample1D();
        values[i] = func(x);
    }
    
    PricingResult result;
    computeStatistics(values, result);
    
    m_totalSamples += numSamples;
    return result;
}

inline Intersection Pricer::intersect(const Ray& ray, const Scene& scene) const {
    Intersection closest;
    
    for (size_t i = 0; i < scene.spheres.size(); i++) {
        float t;
        if (intersectSphere(ray, scene.spheres[i], t)) {
            if (t < closest.t) {
                closest.hit = true;
                closest.t = t;
                closest.point = ray.at(t);
                closest.normal = (closest.point - scene.spheres[i].center).normalized();
                closest.albedo = scene.spheres[i].albedo;
                closest.emission = scene.spheres[i].emission;
                closest.objectIndex = i;
            }
        }
    }
    
    return closest;
}

inline bool Pricer::intersectSphere(const Ray& ray, const Scene::Sphere& sphere, float& t) const {
    Vec3 oc = ray.origin - sphere.center;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0f * oc.dot(ray.direction);
    float c = oc.dot(oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    
    if (discriminant < 0) {
        return false;
    }
    
    float sqrtDisc = std::sqrt(discriminant);
    float t0 = (-b - sqrtDisc) / (2.0f * a);
    float t1 = (-b + sqrtDisc) / (2.0f * a);
    
    if (t0 > 0.001f) {
        t = t0;
        return true;
    }
    if (t1 > 0.001f) {
        t = t1;
        return true;
    }
    
    return false;
}

inline float Pricer::simulateStockPrice(float S0, float r, float sigma, float T) {
    float Z = m_sampler->sample1D();
    // Convert uniform to normal using Box-Muller
    float U1 = m_sampler->sample1D();
    float U2 = m_sampler->sample1D();
    Z = std::sqrt(-2.0f * std::log(U1)) * std::cos(2.0f * M_PI * U2);
    
    float ST = S0 * std::exp((r - 0.5f * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
    return ST;
}

inline std::vector<float> Pricer::simulateStockPath(float S0, float r, float sigma, 
                                                    float T, int numSteps) {
    std::vector<float> path(numSteps + 1);
    path[0] = S0;
    
    float dt = T / numSteps;
    
    for (int i = 1; i <= numSteps; i++) {
        float U1 = m_sampler->sample1D();
        float U2 = m_sampler->sample1D();
        float Z = std::sqrt(-2.0f * std::log(U1)) * std::cos(2.0f * M_PI * U2);
        
        path[i] = path[i-1] * std::exp((r - 0.5f * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
    }
    
    return path;
}

inline void Pricer::computeStatistics(const std::vector<float>& samples, PricingResult& result) {
    result.numSamples = samples.size();
    result.price = mean(samples);
    result.variance = variance(samples, result.price);
    result.standardError = std::sqrt(result.variance / samples.size());
    
    result.minValue = *std::min_element(samples.begin(), samples.end());
    result.maxValue = *std::max_element(samples.begin(), samples.end());
}

inline float Pricer::mean(const std::vector<float>& samples) const {
    float sum = 0.0f;
    for (float s : samples) {
        sum += s;
    }
    return sum / samples.size();
}

inline float Pricer::variance(const std::vector<float>& samples, float mean) const {
    float sumSquared = 0.0f;
    for (float s : samples) {
        float diff = s - mean;
        sumSquared += diff * diff;
    }
    return sumSquared / (samples.size() - 1);
}

} // namespace mc
} // namespace mcgan