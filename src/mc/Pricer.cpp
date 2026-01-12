#include "Pricer.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>

namespace mcgan {
namespace mc {

// ============================================================================
// Constructor
// ============================================================================

Pricer::Pricer(std::shared_ptr<PathSampler> sampler)
    : m_sampler(sampler)
    , m_verbose(false)
    , m_totalSamples(0)
    , m_totalPaths(0)
{
}

// ============================================================================
// Path Tracing Methods
// ============================================================================

Vec3 Pricer::estimateRadiance(const Ray& ray, const Scene& scene, int maxBounces) {
    PathResult result = tracePath(ray, scene, maxBounces);
    return result.radiance;
}

PathResult Pricer::tracePath(const Ray& ray, const Scene& scene, int maxBounces) {
    PathResult result;
    
    Ray currentRay = ray;
    Vec3 throughput(1, 1, 1);
    
    for (int bounce = 0; bounce < maxBounces; bounce++) {
        result.bounces = bounce;
        
        // Intersect scene
        Intersection hit = intersect(currentRay, scene);
        
        if (!hit.hit) {
            // Hit environment/sky - add background color
            result.radiance = result.radiance + throughput * Vec3(0.5f, 0.7f, 1.0f) * 0.1f;
            result.terminated = true;
            break;
        }
        
        // Add emission from light sources
        result.radiance = result.radiance + throughput * hit.emission;
        
        // Russian Roulette for path termination after minimum bounces
        float rrProbability = std::min(1.0f, std::max(throughput.x(), 
                                       std::max(throughput.y(), throughput.z())));
        if (bounce > 3 && !m_sampler->russianRoulette(rrProbability)) {
            result.terminated = true;
            break;
        }
        
        if (bounce > 3) {
            throughput = throughput / rrProbability;
        }
        
        // Sample next direction
        Vec3 nextDir = m_sampler->sampleHemisphere(hit.normal);
        float pdfValue = m_sampler->pdf(nextDir, hit.normal);
        
        if (pdfValue <= 1e-6f) {
            result.terminated = true;
            break;
        }
        
        // Update throughput with BRDF and cosine term
        float cosTheta = std::max(0.0f, nextDir.dot(hit.normal));
        Vec3 brdf = hit.albedo / M_PI;  // Lambertian BRDF
        throughput = throughput * brdf * (cosTheta / pdfValue);
        
        // Prevent numerical issues
        if (throughput.length() < 1e-6f) {
            result.terminated = true;
            break;
        }
        
        // Create next ray with small offset to avoid self-intersection
        currentRay = Ray(hit.point + hit.normal * 0.001f, nextDir);
    }
    
    result.weight = throughput.length();
    m_totalPaths++;
    
    return result;
}

std::vector<Vec3> Pricer::render(const Scene& scene, int width, int height, 
                                 int samplesPerPixel, int maxBounces) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<Vec3> image(width * height, Vec3(0, 0, 0));
    
    float aspectRatio = float(width) / float(height);
    float fov = 60.0f * M_PI / 180.0f;
    float scale = std::tan(fov * 0.5f);
    
    if (m_verbose) {
        std::cout << "Rendering " << width << "x" << height 
                  << " with " << samplesPerPixel << " samples per pixel..." << std::endl;
    }
    
    for (int y = 0; y < height; y++) {
        if (m_verbose && y % 10 == 0) {
            std::cout << "Progress: " << (100 * y / height) << "%" << std::endl;
        }
        
        for (int x = 0; x < width; x++) {
            Vec3 color(0, 0, 0);
            
            for (int s = 0; s < samplesPerPixel; s++) {
                // Pixel coordinates with anti-aliasing jitter
                float px = (2.0f * (x + m_sampler->sample1D()) / width - 1.0f) * aspectRatio * scale;
                float py = (1.0f - 2.0f * (y + m_sampler->sample1D()) / height) * scale;
                
                // Create camera ray
                Vec3 origin(0, 0, 5);  // Camera position
                Vec3 direction(px, py, -1);
                direction = direction.normalized();
                
                Ray ray(origin, direction);
                color = color + estimateRadiance(ray, scene, maxBounces);
            }
            
            // Average over samples
            image[y * width + x] = color / float(samplesPerPixel);
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    if (m_verbose) {
        std::cout << "Rendering completed in " << duration.count() / 1000.0 << " seconds" << std::endl;
    }
    
    return image;
}

// ============================================================================
// Option Pricing Methods
// ============================================================================

PricingResult Pricer::priceEuropeanCall(float S0, float K, float r, float sigma, 
                                        float T, int numSamples) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<float> payoffs(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        float ST = simulateStockPrice(S0, r, sigma, T);
        payoffs[i] = std::max(0.0f, ST - K);
    }
    
    PricingResult result;
    computeStatistics(payoffs, result);
    
    // Discount to present value
    float discountFactor = std::exp(-r * T);
    result.price *= discountFactor;
    result.standardError *= discountFactor;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.computeTime = std::chrono::duration<double>(endTime - startTime).count();
    
    m_totalSamples += numSamples;
    return result;
}

PricingResult Pricer::priceEuropeanPut(float S0, float K, float r, float sigma, 
                                       float T, int numSamples) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<float> payoffs(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        float ST = simulateStockPrice(S0, r, sigma, T);
        payoffs[i] = std::max(0.0f, K - ST);
    }
    
    PricingResult result;
    computeStatistics(payoffs, result);
    
    // Discount to present value
    float discountFactor = std::exp(-r * T);
    result.price *= discountFactor;
    result.standardError *= discountFactor;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.computeTime = std::chrono::duration<double>(endTime - startTime).count();
    
    m_totalSamples += numSamples;
    return result;
}

PricingResult Pricer::priceAsianOption(float S0, float K, float r, float sigma, 
                                       float T, int numSteps, int numSamples, bool isCall) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
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
    
    // Discount to present value
    float discountFactor = std::exp(-r * T);
    result.price *= discountFactor;
    result.standardError *= discountFactor;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.computeTime = std::chrono::duration<double>(endTime - startTime).count();
    
    m_totalSamples += numSamples;
    return result;
}

PricingResult Pricer::pricePathDependent(
    float S0, float r, float sigma, float T, int numSteps, int numSamples,
    std::function<float(const std::vector<float>&)> payoffFunc) {
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<float> payoffs(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        std::vector<float> path = simulateStockPath(S0, r, sigma, T, numSteps);
        payoffs[i] = payoffFunc(path);
    }
    
    PricingResult result;
    computeStatistics(payoffs, result);
    
    // Discount to present value
    float discountFactor = std::exp(-r * T);
    result.price *= discountFactor;
    result.standardError *= discountFactor;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.computeTime = std::chrono::duration<double>(endTime - startTime).count();
    
    m_totalSamples += numSamples;
    return result;
}

// ============================================================================
// General Monte Carlo Methods
// ============================================================================

PricingResult Pricer::estimateIntegral(
    std::function<float(const std::vector<float>&)> integrand,
    int dimension, int numSamples) {
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
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
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.computeTime = std::chrono::duration<double>(endTime - startTime).count();
    
    m_totalSamples += numSamples;
    return result;
}

PricingResult Pricer::estimateExpectation(
    std::function<float(float)> func, int numSamples) {
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<float> values(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        float x = m_sampler->sample1D();
        values[i] = func(x);
    }
    
    PricingResult result;
    computeStatistics(values, result);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.computeTime = std::chrono::duration<double>(endTime - startTime).count();
    
    m_totalSamples += numSamples;
    return result;
}

// ============================================================================
// Scene Intersection Methods
// ============================================================================

Intersection Pricer::intersect(const Ray& ray, const Scene& scene) const {
    Intersection closest;
    closest.t = std::numeric_limits<float>::max();
    
    for (size_t i = 0; i < scene.spheres.size(); i++) {
        float t;
        if (intersectSphere(ray, scene.spheres[i], t)) {
            if (t < closest.t && t > 0.001f) {
                closest.hit = true;
                closest.t = t;
                closest.point = ray.at(t);
                closest.normal = (closest.point - scene.spheres[i].center).normalized();
                closest.albedo = scene.spheres[i].albedo;
                closest.emission = scene.spheres[i].emission;
                closest.objectIndex = static_cast<int>(i);
            }
        }
    }
    
    return closest;
}

bool Pricer::intersectSphere(const Ray& ray, const Scene::Sphere& sphere, float& t) const {
    Vec3 oc = ray.origin - sphere.center;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0f * oc.dot(ray.direction);
    float c = oc.dot(oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4.0f * a * c;
    
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

Vec3 Pricer::computeDirectLighting(const Intersection& hit, const Scene& scene) {
    Vec3 directLight(0, 0, 0);
    
    // Sample light direction
    Vec3 lightDir = (scene.lightPosition - hit.point).normalized();
    float lightDistance = (scene.lightPosition - hit.point).length();
    
    // Check if light is visible (shadow ray)
    Ray shadowRay(hit.point + hit.normal * 0.001f, lightDir);
    Intersection shadowHit = intersect(shadowRay, scene);
    
    if (!shadowHit.hit || shadowHit.t > lightDistance) {
        // Light is visible
        float cosTheta = std::max(0.0f, hit.normal.dot(lightDir));
        float attenuation = 1.0f / (lightDistance * lightDistance);
        
        Vec3 brdf = hit.albedo / M_PI;
        directLight = scene.lightColor * scene.lightIntensity * brdf * cosTheta * attenuation;
    }
    
    return directLight;
}

// ============================================================================
// Stock Price Simulation Methods
// ============================================================================

float Pricer::simulateStockPrice(float S0, float r, float sigma, float T) {
    // Box-Muller transform for normal distribution
    float U1 = m_sampler->sample1D();
    float U2 = m_sampler->sample1D();
    
    // Ensure U1 is not zero to avoid log(0)
    U1 = std::max(U1, 1e-10f);
    
    float Z = std::sqrt(-2.0f * std::log(U1)) * std::cos(2.0f * M_PI * U2);
    
    // Geometric Brownian Motion
    float drift = (r - 0.5f * sigma * sigma) * T;
    float diffusion = sigma * std::sqrt(T) * Z;
    float ST = S0 * std::exp(drift + diffusion);
    
    return ST;
}

std::vector<float> Pricer::simulateStockPath(float S0, float r, float sigma, 
                                             float T, int numSteps) {
    std::vector<float> path(numSteps + 1);
    path[0] = S0;
    
    float dt = T / numSteps;
    float sqrtDt = std::sqrt(dt);
    
    for (int i = 1; i <= numSteps; i++) {
        // Box-Muller for normal random variable
        float U1 = m_sampler->sample1D();
        float U2 = m_sampler->sample1D();
        
        U1 = std::max(U1, 1e-10f);
        
        float Z = std::sqrt(-2.0f * std::log(U1)) * std::cos(2.0f * M_PI * U2);
        
        // Update stock price
        float drift = (r - 0.5f * sigma * sigma) * dt;
        float diffusion = sigma * sqrtDt * Z;
        path[i] = path[i-1] * std::exp(drift + diffusion);
    }
    
    return path;
}

// ============================================================================
// Statistical Utility Methods
// ============================================================================

void Pricer::computeStatistics(const std::vector<float>& samples, PricingResult& result) {
    result.numSamples = static_cast<int>(samples.size());
    
    if (samples.empty()) {
        result.price = 0.0f;
        result.variance = 0.0f;
        result.standardError = 0.0f;
        result.minValue = 0.0f;
        result.maxValue = 0.0f;
        return;
    }
    
    result.price = mean(samples);
    result.variance = variance(samples, result.price);
    result.standardError = std::sqrt(result.variance / samples.size());
    
    result.minValue = *std::min_element(samples.begin(), samples.end());
    result.maxValue = *std::max_element(samples.begin(), samples.end());
}

float Pricer::mean(const std::vector<float>& samples) const {
    if (samples.empty()) return 0.0f;
    
    float sum = 0.0f;
    for (float s : samples) {
        sum += s;
    }
    return sum / samples.size();
}

float Pricer::variance(const std::vector<float>& samples, float sampleMean) const {
    if (samples.size() <= 1) return 0.0f;
    
    float sumSquared = 0.0f;
    for (float s : samples) {
        float diff = s - sampleMean;
        sumSquared += diff * diff;
    }
    return sumSquared / (samples.size() - 1);
}

} // namespace mc
} // namespace mcgan