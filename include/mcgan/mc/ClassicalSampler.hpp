#pragma once

#include "PathSampler.hpp"
#include "../core/Types.hpp"
#include <random>
#include <memory>

namespace mcgan {
namespace mc {

/**
 * Classical Monte Carlo sampler using uniform random sampling
 * for path tracing and light transport simulation.
 */
class ClassicalSampler : public PathSampler {
public:
    ClassicalSampler(int seed = 0);
    virtual ~ClassicalSampler() = default;

    // Sample a direction in the hemisphere (cosine-weighted)
    virtual Vec3 sampleHemisphere(const Vec3& normal) override;
    
    // Sample a point on a light source
    virtual Vec3 sampleLight(const Vec3& position) override;
    
    // Generate a random value in [0, 1)
    virtual float sample1D() override;
    
    // Generate a 2D random sample in [0, 1)^2
    virtual Vec2 sample2D() override;
    
    // Compute the probability density function for a sampled direction
    virtual float pdf(const Vec3& direction, const Vec3& normal) override;
    
    // Reset the sampler with a new seed
    virtual void reset(int seed) override;
    
    // Get the current sample index
    virtual int getSampleIndex() const override { return m_sampleIndex; }
    
    // Advance to the next sample
    virtual void nextSample() override { m_sampleIndex++; }

protected:
    std::mt19937 m_rng;
    std::uniform_real_distribution<float> m_dist;
    int m_sampleIndex;
    int m_seed;

    // Helper function for cosine-weighted hemisphere sampling
    Vec3 cosineWeightedSample(float u1, float u2, const Vec3& normal);
    
    // Create orthonormal basis from a normal vector
    void createCoordinateSystem(const Vec3& N, Vec3& Nt, Vec3& Nb);
};

// Implementation
inline ClassicalSampler::ClassicalSampler(int seed)
    : m_rng(seed)
    , m_dist(0.0f, 1.0f)
    , m_sampleIndex(0)
    , m_seed(seed)
{
}

inline Vec3 ClassicalSampler::sampleHemisphere(const Vec3& normal) {
    float u1 = sample1D();
    float u2 = sample1D();
    return cosineWeightedSample(u1, u2, normal);
}

inline Vec3 ClassicalSampler::sampleLight(const Vec3& position) {
    // Uniform sampling on a sphere for simple light sampling
    float u1 = sample1D();
    float u2 = sample1D();
    
    float z = 1.0f - 2.0f * u1;
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = 2.0f * M_PI * u2;
    
    return Vec3(r * std::cos(phi), r * std::sin(phi), z);
}

inline float ClassicalSampler::sample1D() {
    return m_dist(m_rng);
}

inline Vec2 ClassicalSampler::sample2D() {
    return Vec2(sample1D(), sample1D());
}

inline float ClassicalSampler::pdf(const Vec3& direction, const Vec3& normal) {
    // Cosine-weighted PDF: cos(theta) / pi
    float cosTheta = std::max(0.0f, direction.dot(normal));
    return cosTheta / M_PI;
}

inline void ClassicalSampler::reset(int seed) {
    m_seed = seed;
    m_rng.seed(seed);
    m_sampleIndex = 0;
}

inline Vec3 ClassicalSampler::cosineWeightedSample(float u1, float u2, const Vec3& normal) {
    // Cosine-weighted hemisphere sampling using Malley's method
    float r = std::sqrt(u1);
    float theta = 2.0f * M_PI * u2;
    
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1));
    
    // Create local coordinate system
    Vec3 Nt, Nb;
    createCoordinateSystem(normal, Nt, Nb);
    
    // Transform to world space
    return (Nt * x + Nb * y + normal * z).normalized();
}

inline void ClassicalSampler::createCoordinateSystem(const Vec3& N, Vec3& Nt, Vec3& Nb) {
    if (std::abs(N.x()) > std::abs(N.y())) {
        Nt = Vec3(N.z(), 0, -N.x()).normalized();
    } else {
        Nt = Vec3(0, -N.z(), N.y()).normalized();
    }
    Nb = N.cross(Nt);
}

} // namespace mc
} // namespace mcgan