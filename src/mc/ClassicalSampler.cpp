#include "ClassicalSampler.hpp"
#include <cmath>

namespace mcgan {
namespace mc {

ClassicalSampler::ClassicalSampler(int seed)
    : m_rng(seed)
    , m_dist(0.0f, 1.0f)
    , m_sampleIndex(0)
    , m_seed(seed)
{
}

Vec3 ClassicalSampler::sampleHemisphere(const Vec3& normal) {
    float u1 = sample1D();
    float u2 = sample1D();
    return cosineWeightedSample(u1, u2, normal);
}

Vec3 ClassicalSampler::sampleLight(const Vec3& position) {
    // Uniform sampling on a sphere for simple light sampling
    float u1 = sample1D();
    float u2 = sample1D();
    
    float z = 1.0f - 2.0f * u1;
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = 2.0f * M_PI * u2;
    
    return Vec3(r * std::cos(phi), r * std::sin(phi), z);
}

float ClassicalSampler::sample1D() {
    return m_dist(m_rng);
}

Vec2 ClassicalSampler::sample2D() {
    return Vec2(sample1D(), sample1D());
}

float ClassicalSampler::pdf(const Vec3& direction, const Vec3& normal) {
    // Cosine-weighted PDF: cos(theta) / pi
    float cosTheta = std::max(0.0f, direction.dot(normal));
    return cosTheta / M_PI;
}

void ClassicalSampler::reset(int seed) {
    m_seed = seed;
    m_rng.seed(seed);
    m_sampleIndex = 0;
}

Vec3 ClassicalSampler::cosineWeightedSample(float u1, float u2, const Vec3& normal) {
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

void ClassicalSampler::createCoordinateSystem(const Vec3& N, Vec3& Nt, Vec3& Nb) {
    if (std::abs(N.x()) > std::abs(N.y())) {
        Nt = Vec3(N.z(), 0, -N.x()).normalized();
    } else {
        Nt = Vec3(0, -N.z(), N.y()).normalized();
    }
    Nb = N.cross(Nt);
}

} // namespace mc
} // namespace mcgan