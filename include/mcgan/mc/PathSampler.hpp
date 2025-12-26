#pragma once

#include "../core/Types.hpp"
#include <memory>

namespace mcgan {
namespace mc {

/**
 * Abstract base class for path sampling strategies in Monte Carlo rendering.
 * Defines the interface for different sampling methods including classical
 * Monte Carlo and GAN-based importance sampling.
 */
class PathSampler {
public:
    virtual ~PathSampler() = default;

    /**
     * Sample a direction in the hemisphere around a given normal.
     * @param normal The surface normal vector (should be normalized)
     * @return A sampled direction in the hemisphere
     */
    virtual Vec3 sampleHemisphere(const Vec3& normal) = 0;

    /**
     * Sample a point or direction toward a light source.
     * @param position The position from which to sample
     * @return A sampled light direction or position
     */
    virtual Vec3 sampleLight(const Vec3& position) = 0;

    /**
     * Generate a random sample in the range [0, 1).
     * @return A random float value
     */
    virtual float sample1D() = 0;

    /**
     * Generate a 2D random sample in the range [0, 1)^2.
     * @return A random 2D vector
     */
    virtual Vec2 sample2D() = 0;

    /**
     * Calculate the probability density function (PDF) for a sampled direction.
     * @param direction The sampled direction
     * @param normal The surface normal
     * @return The PDF value for the given direction
     */
    virtual float pdf(const Vec3& direction, const Vec3& normal) = 0;

    /**
     * Reset the sampler to a new state.
     * @param seed The random seed for reproducibility
     */
    virtual void reset(int seed) = 0;

    /**
     * Get the current sample index.
     * @return The current sample number
     */
    virtual int getSampleIndex() const = 0;

    /**
     * Advance to the next sample in the sequence.
     */
    virtual void nextSample() = 0;

    /**
     * Sample a direction using Russian Roulette for path termination.
     * @param throughput The current path throughput
     * @param minThroughput Minimum throughput before applying RR
     * @return true if the path should continue, false if terminated
     */
    virtual bool russianRoulette(float throughput, float minThroughput = 0.1f) {
        if (throughput >= minThroughput) {
            return true;
        }
        
        float q = std::max(0.05f, 1.0f - throughput);
        return sample1D() >= q;
    }

    /**
     * Compute the Russian Roulette continuation probability.
     * @param throughput The current path throughput
     * @param minThroughput Minimum throughput before applying RR
     * @return The probability of continuing the path
     */
    virtual float getRussianRouletteProbability(float throughput, float minThroughput = 0.1f) const {
        if (throughput >= minThroughput) {
            return 1.0f;
        }
        
        float q = std::max(0.05f, 1.0f - throughput);
        return 1.0f - q;
    }

    /**
     * Sample multiple directions at once (for batch processing).
     * @param normal The surface normal
     * @param count Number of directions to sample
     * @return Vector of sampled directions
     */
    virtual std::vector<Vec3> sampleHemisphereBatch(const Vec3& normal, int count) {
        std::vector<Vec3> samples(count);
        for (int i = 0; i < count; i++) {
            samples[i] = sampleHemisphere(normal);
        }
        return samples;
    }

    /**
     * Evaluate the BRDF for a given pair of directions.
     * This is a simple Lambertian BRDF implementation.
     * @param wi Incident direction
     * @param wo Outgoing direction
     * @param normal Surface normal
     * @return BRDF value
     */
    virtual Vec3 evaluateBRDF(const Vec3& wi, const Vec3& wo, const Vec3& normal) const {
        // Simple Lambertian BRDF: albedo / pi
        // Assuming white albedo for base implementation
        float cosTheta = std::max(0.0f, wo.dot(normal));
        if (cosTheta > 0.0f) {
            return Vec3(1.0f / M_PI, 1.0f / M_PI, 1.0f / M_PI);
        }
        return Vec3(0, 0, 0);
    }

    /**
     * Compute importance sampling weight (BRDF * cosTheta / PDF).
     * @param direction The sampled direction
     * @param normal The surface normal
     * @return The importance weight
     */
    virtual float computeWeight(const Vec3& direction, const Vec3& normal) {
        float cosTheta = std::max(0.0f, direction.dot(normal));
        float pdfValue = pdf(direction, normal);
        
        if (pdfValue <= 0.0f) {
            return 0.0f;
        }
        
        // For Lambertian: (albedo/pi * cosTheta) / pdf
        // With albedo=1: cosTheta / (pi * pdf)
        return cosTheta / (M_PI * pdfValue);
    }

protected:
    /**
     * Clamp a value to a valid range.
     */
    float clamp(float value, float min, float max) const {
        return std::max(min, std::min(max, value));
    }

    /**
     * Convert a 2D uniform sample to a direction on the unit hemisphere.
     */
    Vec3 uniformSampleHemisphere(float u1, float u2) const {
        float z = u1;
        float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        float phi = 2.0f * M_PI * u2;
        return Vec3(r * std::cos(phi), r * std::sin(phi), z);
    }

    /**
     * Convert a 2D uniform sample to a cosine-weighted direction.
     */
    Vec3 cosineSampleHemisphere(float u1, float u2) const {
        float r = std::sqrt(u1);
        float theta = 2.0f * M_PI * u2;
        float x = r * std::cos(theta);
        float y = r * std::sin(theta);
        float z = std::sqrt(std::max(0.0f, 1.0f - u1));
        return Vec3(x, y, z);
    }
};

/**
 * Factory function to create samplers.
 */
inline std::shared_ptr<PathSampler> createSampler(const std::string& type, int seed = 0) {
    // This would typically be implemented in a .cpp file
    // with proper instantiation of ClassicalSampler or GanPathSampler
    return nullptr;
}

} // namespace mc
} // namespace mcgan