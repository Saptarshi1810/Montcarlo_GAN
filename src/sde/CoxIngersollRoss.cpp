#include "CoxIngersollRoss.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mcgan {
namespace sde {

// ============================================================================
// CoxIngersollRoss Implementation
// ============================================================================

// Note: Most methods are inline in the header for performance.
// This file serves as a placeholder for any additional implementations.

// Helper function to validate CIR parameters
bool CoxIngersollRoss::satisfiesFellerCondition() const {
    return 2.0f * m_kappa * m_theta >= m_sigma * m_sigma;
}

float CoxIngersollRoss::getLongTermVariance() const {
    if (m_kappa <= 0.0f) {
        throw std::runtime_error("Kappa must be positive");
    }
    return m_sigma * m_sigma * m_theta / (2.0f * m_kappa);
}

float CoxIngersollRoss::expectedValue(float r0, float t) const {
    // E[r_t | r_0] = r_0 * e^(-κt) + θ(1 - e^(-κt))
    float expTerm = std::exp(-m_kappa * t);
    return r0 * expTerm + m_theta * (1.0f - expTerm);
}

float CoxIngersollRoss::variance(float r0, float t) const {
    // Var[r_t | r_0]
    float expTerm = std::exp(-m_kappa * t);
    float exp2Term = std::exp(-2.0f * m_kappa * t);
    
    float term1 = r0 * m_sigma * m_sigma / m_kappa * (expTerm - exp2Term);
    float term2 = m_theta * m_sigma * m_sigma / (2.0f * m_kappa) * 
                  (1.0f - expTerm) * (1.0f - expTerm);
    
    return term1 + term2;
}

std::vector<float> CoxIngersollRoss::drift(const std::vector<float>& state, float t) const {
    // κ(θ - r_t)
    float r = std::max(0.0f, state[0]);  // Ensure non-negative
    return {m_kappa * (m_theta - r)};
}

std::vector<float> CoxIngersollRoss::diffusion(const std::vector<float>& state, float t) const {
    // σ√r_t
    float r = std::max(0.0f, state[0]);  // Ensure non-negative
    return {m_sigma * std::sqrt(r)};
}

// ============================================================================
// CIRTimeDep Implementation
// ============================================================================

std::vector<float> CIRTimeDep::drift(const std::vector<float>& state, float t) const {
    float r = std::max(0.0f, state[0]);
    float kappa = m_kappaFunc(t);
    float theta = m_thetaFunc(t);
    return {kappa * (theta - r)};
}

std::vector<float> CIRTimeDep::diffusion(const std::vector<float>& state, float t) const {
    float r = std::max(0.0f, state[0]);
    float sigma = m_sigmaFunc(t);
    return {sigma * std::sqrt(r)};
}

// ============================================================================
// MultiFactorCIR Implementation
// ============================================================================

std::vector<float> MultiFactorCIR::drift(const std::vector<float>& state, float t) const {
    std::vector<float> result(m_dimension);
    
    for (int i = 0; i < m_dimension; i++) {
        float r = std::max(0.0f, state[i]);
        result[i] = m_factors[i].kappa * (m_factors[i].theta - r);
    }
    
    return result;
}

std::vector<float> MultiFactorCIR::diffusion(const std::vector<float>& state, float t) const {
    // Return diagonal diffusion matrix
    std::vector<float> result(m_dimension * m_dimension, 0.0f);
    
    for (int i = 0; i < m_dimension; i++) {
        float r = std::max(0.0f, state[i]);
        result[i * m_dimension + i] = m_factors[i].sigma * std::sqrt(r);
    }
    
    return result;
}

std::vector<float> MultiFactorCIR::getParameters() const {
    std::vector<float> params;
    params.reserve(m_factors.size() * 3);
    
    for (const auto& factor : m_factors) {
        params.push_back(factor.kappa);
        params.push_back(factor.theta);
        params.push_back(factor.sigma);
    }
    
    return params;
}

void MultiFactorCIR::setParameters(const std::vector<float>& params) {
    size_t idx = 0;
    
    for (size_t i = 0; i < m_factors.size() && idx + 2 < params.size(); i++) {
        m_factors[i].kappa = params[idx++];
        m_factors[i].theta = params[idx++];
        m_factors[i].sigma = params[idx++];
    }
}

} // namespace sde
} // namespace mcgan