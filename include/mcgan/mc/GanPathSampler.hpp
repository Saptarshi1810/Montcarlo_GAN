#pragma once

#include "PathSampler.hpp"
#include "../core/Types.hpp"
#include "../nn/Network.hpp"
#include <memory>
#include <vector>

namespace mcgan {
namespace mc {

/**
 * GAN-based path sampler that uses a neural network to generate
 * importance-sampled directions for path tracing.
 * This reduces variance compared to classical Monte Carlo methods.
 */
class GanPathSampler : public PathSampler {
public:
    GanPathSampler(std::shared_ptr<nn::Network> generator, int seed = 0);
    virtual ~GanPathSampler() = default;

    // Sample a direction using the GAN generator
    virtual Vec3 sampleHemisphere(const Vec3& normal) override;
    
    // Sample a light direction using the network
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
    
    // Set the scene context for conditional sampling
    void setSceneContext(const std::vector<float>& context);
    
    // Enable/disable GAN sampling (fallback to classical if disabled)
    void setGanEnabled(bool enabled) { m_ganEnabled = enabled; }
    bool isGanEnabled() const { return m_ganEnabled; }
    
    // Get statistics
    int getNumGanSamples() const { return m_numGanSamples; }
    int getNumFallbackSamples() const { return m_numFallbackSamples; }

protected:
    std::shared_ptr<nn::Network> m_generator;
    std::mt19937 m_rng;
    std::uniform_real_distribution<float> m_dist;
    std::normal_distribution<float> m_normalDist;
    
    int m_sampleIndex;
    int m_seed;
    bool m_ganEnabled;
    
    // Statistics
    int m_numGanSamples;
    int m_numFallbackSamples;
    
    // Scene context for conditional generation
    std::vector<float> m_sceneContext;
    
    // Latent space dimension for the generator
    int m_latentDim;
    
    // Helper functions
    Vec3 ganSample(const Vec3& normal);
    Vec3 classicalSample(const Vec3& normal);
    std::vector<float> generateLatentVector();
    Vec3 networkOutputToDirection(const std::vector<float>& output, const Vec3& normal);
    void createCoordinateSystem(const Vec3& N, Vec3& Nt, Vec3& Nb);
    
    // PDF estimation for GAN samples
    float estimateGanPdf(const Vec3& direction, const Vec3& normal);
};

// Implementation
inline GanPathSampler::GanPathSampler(std::shared_ptr<nn::Network> generator, int seed)
    : m_generator(generator)
    , m_rng(seed)
    , m_dist(0.0f, 1.0f)
    , m_normalDist(0.0f, 1.0f)
    , m_sampleIndex(0)
    , m_seed(seed)
    , m_ganEnabled(true)
    , m_numGanSamples(0)
    , m_numFallbackSamples(0)
    , m_latentDim(128)  // Default latent dimension
{
}

inline Vec3 GanPathSampler::sampleHemisphere(const Vec3& normal) {
    if (m_ganEnabled && m_generator) {
        m_numGanSamples++;
        return ganSample(normal);
    } else {
        m_numFallbackSamples++;
        return classicalSample(normal);
    }
}

inline Vec3 GanPathSampler::sampleLight(const Vec3& position) {
    // Use GAN for light sampling if available
    if (m_ganEnabled && m_generator) {
        Vec3 up(0, 1, 0);
        return ganSample(up);
    }
    
    // Fallback to uniform sphere sampling
    float u1 = sample1D();
    float u2 = sample1D();
    
    float z = 1.0f - 2.0f * u1;
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = 2.0f * M_PI * u2;
    
    return Vec3(r * std::cos(phi), r * std::sin(phi), z);
}

inline float GanPathSampler::sample1D() {
    return m_dist(m_rng);
}

inline Vec2 GanPathSampler::sample2D() {
    return Vec2(sample1D(), sample1D());
}

inline float GanPathSampler::pdf(const Vec3& direction, const Vec3& normal) {
    if (m_ganEnabled && m_generator) {
        return estimateGanPdf(direction, normal);
    }
    
    // Fallback to cosine-weighted PDF
    float cosTheta = std::max(0.0f, direction.dot(normal));
    return cosTheta / M_PI;
}

inline void GanPathSampler::reset(int seed) {
    m_seed = seed;
    m_rng.seed(seed);
    m_sampleIndex = 0;
    m_numGanSamples = 0;
    m_numFallbackSamples = 0;
}

inline void GanPathSampler::setSceneContext(const std::vector<float>& context) {
    m_sceneContext = context;
}

inline Vec3 GanPathSampler::ganSample(const Vec3& normal) {
    // Generate latent vector
    std::vector<float> latent = generateLatentVector();
    
    // Concatenate with scene context if available
    if (!m_sceneContext.empty()) {
        latent.insert(latent.end(), m_sceneContext.begin(), m_sceneContext.end());
    }
    
    // Run through generator network
    std::vector<float> output = m_generator->forward(latent);
    
    // Convert network output to direction vector
    return networkOutputToDirection(output, normal);
}

inline Vec3 GanPathSampler::classicalSample(const Vec3& normal) {
    float u1 = sample1D();
    float u2 = sample1D();
    
    // Cosine-weighted hemisphere sampling
    float r = std::sqrt(u1);
    float theta = 2.0f * M_PI * u2;
    
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1));
    
    Vec3 Nt, Nb;
    createCoordinateSystem(normal, Nt, Nb);
    
    return (Nt * x + Nb * y + normal * z).normalized();
}

inline std::vector<float> GanPathSampler::generateLatentVector() {
    std::vector<float> latent(m_latentDim);
    for (int i = 0; i < m_latentDim; i++) {
        latent[i] = m_normalDist(m_rng);
    }
    return latent;
}

inline Vec3 GanPathSampler::networkOutputToDirection(const std::vector<float>& output, const Vec3& normal) {
    // Network outputs spherical coordinates or cartesian direction
    // Assume output has at least 3 values
    if (output.size() >= 3) {
        Vec3 direction(output[0], output[1], output[2]);
        direction = direction.normalized();
        
        // Ensure direction is in the hemisphere
        if (direction.dot(normal) < 0) {
            direction = -direction;
        }
        
        return direction;
    }
    
    // Fallback: convert 2D output to spherical coordinates
    float theta = std::acos(std::clamp(output[0], -1.0f, 1.0f));
    float phi = output.size() > 1 ? output[1] * 2.0f * M_PI : 0.0f;
    
    float sinTheta = std::sin(theta);
    Vec3 localDir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), std::cos(theta));
    
    Vec3 Nt, Nb;
    createCoordinateSystem(normal, Nt, Nb);
    
    return (Nt * localDir.x() + Nb * localDir.y() + normal * localDir.z()).normalized();
}

inline void GanPathSampler::createCoordinateSystem(const Vec3& N, Vec3& Nt, Vec3& Nb) {
    if (std::abs(N.x()) > std::abs(N.y())) {
        Nt = Vec3(N.z(), 0, -N.x()).normalized();
    } else {
        Nt = Vec3(0, -N.z(), N.y()).normalized();
    }
    Nb = N.cross(Nt);
}

inline float GanPathSampler::estimateGanPdf(const Vec3& direction, const Vec3& normal) {
    // Estimate PDF using Monte Carlo with multiple samples
    // This is a simplified approximation
    // In practice, you might want to train a discriminator to estimate density
    
    // For now, use a mixture of cosine-weighted and uniform
    float cosTheta = std::max(0.0f, direction.dot(normal));
    float cosinePdf = cosTheta / M_PI;
    float uniformPdf = 1.0f / (2.0f * M_PI);
    
    // Weighted mixture
    return 0.8f * cosinePdf + 0.2f * uniformPdf;
}

} // namespace mc
} // namespace mcgan