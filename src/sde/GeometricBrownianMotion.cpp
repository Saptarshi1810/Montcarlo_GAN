#include "GeometricBrownianMotion.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace mcgan {
namespace sde {

// ============================================================================
// GeometricBrownianMotion Implementation
// ============================================================================

std::vector<float> GeometricBrownianMotion::drift(
    const std::vector<float>& state, float t) const {
    // μ S_t
    return {m_mu * state[0]};
}

std::vector<float> GeometricBrownianMotion::diffusion(
    const std::vector<float>& state, float t) const {
    // σ S_t
    return {m_sigma * state[0]};
}

float GeometricBrownianMotion::exactSolution(float S0, float t, float Z) const {
    // S_t = S_0 * exp((μ - σ²/2)t + σ√t Z)
    float drift = (m_mu - 0.5f * m_sigma * m_sigma) * t;
    float diffusion = m_sigma * std::sqrt(t) * Z;
    return S0 * std::exp(drift + diffusion);
}

// ============================================================================
// MultiAssetGBM Implementation
// ============================================================================

MultiAssetGBM::MultiAssetGBM(const std::vector<float>& mu, 
                             const std::vector<float>& sigma,
                             const std::vector<float>& correlation)
    : m_mu(mu)
    , m_sigma(sigma)
    , m_correlation(correlation)
    , m_dimension(static_cast<int>(mu.size()))
{
    if (mu.size() != sigma.size()) {
        throw std::invalid_argument("Mu and sigma must have the same size");
    }
    
    if (m_correlation.empty()) {
        // Create identity correlation matrix
        m_correlation.resize(m_dimension * m_dimension, 0.0f);
        for (int i = 0; i < m_dimension; i++) {
            m_correlation[i * m_dimension + i] = 1.0f;
        }
    } else {
        // Validate correlation matrix size
        if (m_correlation.size() != static_cast<size_t>(m_dimension * m_dimension)) {
            throw std::invalid_argument("Correlation matrix size mismatch");
        }
    }
    
    // Compute Cholesky decomposition
    computeCholesky();
}

std::vector<float> MultiAssetGBM::drift(const std::vector<float>& state, float t) const {
    std::vector<float> result(m_dimension);
    
    for (int i = 0; i < m_dimension; i++) {
        result[i] = m_mu[i] * state[i];
    }
    
    return result;
}

std::vector<float> MultiAssetGBM::diffusion(const std::vector<float>& state, float t) const {
    // Return diagonal volatility matrix scaled by state
    std::vector<float> result(m_dimension * m_dimension, 0.0f);
    
    for (int i = 0; i < m_dimension; i++) {
        // Diagonal elements: σ_i * S_i
        result[i * m_dimension + i] = m_sigma[i] * state[i];
    }
    
    return result;
}

std::vector<float> MultiAssetGBM::getParameters() const {
    std::vector<float> params;
    params.reserve(2 * m_dimension);
    
    params.insert(params.end(), m_mu.begin(), m_mu.end());
    params.insert(params.end(), m_sigma.begin(), m_sigma.end());
    
    return params;
}

void MultiAssetGBM::setParameters(const std::vector<float>& params) {
    int n = m_dimension;
    
    if (params.size() >= static_cast<size_t>(n)) {
        m_mu.assign(params.begin(), params.begin() + n);
    }
    
    if (params.size() >= static_cast<size_t>(2 * n)) {
        m_sigma.assign(params.begin() + n, params.begin() + 2 * n);
    }
}

std::vector<float> MultiAssetGBM::generateCorrelatedNormals(
    const std::vector<float>& independent) const {
    
    if (independent.size() != static_cast<size_t>(m_dimension)) {
        throw std::invalid_argument("Independent vector size mismatch");
    }
    
    return SdeUtils::matmul(m_cholesky, independent, m_dimension, m_dimension);
}

void MultiAssetGBM::computeCholesky() {
    // Compute Cholesky decomposition: L such that LL^T = Correlation
    int n = m_dimension;
    m_cholesky.resize(n * n, 0.0f);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            
            if (j == i) {
                // Diagonal element
                for (int k = 0; k < j; k++) {
                    sum += m_cholesky[j * n + k] * m_cholesky[j * n + k];
                }
                
                float diag = m_correlation[j * n + j] - sum;
                if (diag < 0.0f) {
                    throw std::runtime_error("Correlation matrix is not positive definite");
                }
                
                m_cholesky[j * n + j] = std::sqrt(diag);
            } else {
                // Off-diagonal element
                for (int k = 0; k < j; k++) {
                    sum += m_cholesky[i * n + k] * m_cholesky[j * n + k];
                }
                
                if (std::abs(m_cholesky[j * n + j]) < 1e-10f) {
                    throw std::runtime_error("Cholesky decomposition failed: zero diagonal");
                }
                
                m_cholesky[i * n + j] = (m_correlation[i * n + j] - sum) / m_cholesky[j * n + j];
            }
        }
    }
}

} // namespace sde
} // namespace mcgan