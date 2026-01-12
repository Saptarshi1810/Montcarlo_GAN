#include "GanPathSampler.hpp"
#include <cmath>
#include <algorithm>

namespace mcgan {
namespace mc {

GanPathSampler::GanPathSampler(std::shared_ptr<nn::Network> generator, int seed)
    : m_generator(generator)
    , m_rng(seed)
    , m_dist(0.0f, 1.0f)
    , m_normalDist(0.0f, 1.0f)
    , m_sampleIndex(0)
    , m_seed(seed)
    , m_ganEnabled(true)
    , m_numGanSamples(0)
    , m_numFallbackSamples(0)
    , m_latentDim(128)
{
}

Vec3 GanPathSampler::sampleHemisphere(const Vec3& normal) {
    if (m_ganEnabled && m_generator) {
        m_numGanSamples++;
        return ganSample(normal);
    } else {
        m_numFallbackSamples++;
        return classicalSample(normal);
    }
}

Vec3 GanPathSampler::sampleLight(const Vec3& position) {
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

float GanPathSampler::sample1D() {
    return m_dist(m_rng);
}

Vec2 GanPathSampler::sample2D() {
    return Vec2(sample1D(), sample1D());
}

float GanPathSampler::pdf(const Vec3& direction, const Vec3& normal) {
    if (m_ganEnabled && m_generator) {
        return estimateGanPdf(direction, normal);
    }
    
    // Fallback to cosine-weighted PDF
    float cosTheta = std::max(0.0f, direction.dot(normal));
    return cosTheta / M_PI;
}

void GanPathSampler::reset(int seed) {
    m_seed = seed;
    m_rng.seed(seed);
    m_sampleIndex = 0;
    m_numGanSamples = 0;
    m_numFallbackSamples = 0;
}

void GanPathSampler::setSceneContext(const std::vector<float>& context) {
    m_sceneContext = context;
}

Vec3 GanPathSampler::ganSample(const Vec3& normal) {
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

Vec3 GanPathSampler::classicalSample(const Vec3& normal) {
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

std::vector<float> GanPathSampler::generateLatentVector() {
    std::vector<float> latent(m_latentDim);
    for (int i = 0; i < m_latentDim; i++) {
        latent[i] = m_normalDist(m_rng);
    }
    return latent;
}

Vec3 GanPathSampler::networkOutputToDirection(const std::vector<float>& output, const Vec3& normal) {
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

void GanPathSampler::createCoordinateSystem(const Vec3& N, Vec3& Nt, Vec3& Nb) {
    if (std::abs(N.x()) > std::abs(N.y())) {
        Nt = Vec3(N.z(), 0, -N.x()).normalized();
    } else {
        Nt = Vec3(0, -N.z(), N.y()).normalized();
    }
    Nb = N.cross(Nt);
}

float GanPathSampler::estimateGanPdf(const Vec3& direction, const Vec3& normal) {
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