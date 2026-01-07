#pragma once

#include "SdeModel.hpp"
#include <cmath>

namespace mcgan {
namespace sde {

/**
 * Geometric Brownian Motion (GBM) model.
 * 
 * Used extensively in financial modeling (Black-Scholes).
 * dS_t = μ S_t dt + σ S_t dW_t
 * 
 * where:
 *   S_t = asset price at time t
 *   μ = drift (expected return)
 *   σ = volatility (standard deviation of returns)
 *   W_t = Wiener process (Brownian motion)
 */
class GeometricBrownianMotion : public SdeModel {
public:
    /**
     * Constructor.
     * @param mu Drift coefficient (expected return)
     * @param sigma Volatility coefficient
     */
    GeometricBrownianMotion(float mu = 0.05f, float sigma = 0.2f)
        : m_mu(mu)
        , m_sigma(sigma)
    {}
    
    virtual std::vector<float> drift(const std::vector<float>& state, float t) const override {
        // μ S_t
        return {m_mu * state[0]};
    }
    
    virtual std::vector<float> diffusion(const std::vector<float>& state, float t) const override {
        // σ S_t
        return {m_sigma * state[0]};
    }
    
    virtual int getDimension() const override { return 1; }
    
    virtual std::string getName() const override { 
        return "Geometric Brownian Motion"; 
    }
    
    virtual bool hasConstantDiffusion() const override { return false; }
    
    virtual std::vector<float> getParameters() const override {
        return {m_mu, m_sigma};
    }
    
    virtual void setParameters(const std::vector<float>& params) override {
        if (params.size() >= 1) m_mu = params[0];
        if (params.size() >= 2) m_sigma = params[1];
    }
    
    // Analytical solution (exact simulation)
    float exactSolution(float S0, float t, float Z) const {
        // S_t = S_0 * exp((μ - σ²/2)t + σ√t Z)
        return S0 * std::exp((m_mu - 0.5f * m_sigma * m_sigma) * t + 
                            m_sigma * std::sqrt(t) * Z);
    }
    
    // Getters
    float getMu() const { return m_mu; }
    float getSigma() const { return m_sigma; }
    
    // Setters
    void setMu(float mu) { m_mu = mu; }
    void setSigma(float sigma) { m_sigma = sigma; }

private:
    float m_mu;     // Drift
    float m_sigma;  // Volatility
};

/**
 * Multi-asset Geometric Brownian Motion.
 * 
 * For simulating multiple correlated assets:
 * dS_i = μ_i S_i dt + σ_i S_i dW_i
 * where dW_i dW_j = ρ_{ij} dt
 */
class MultiAssetGBM : public SdeModel {
public:
    /**
     * Constructor.
     * @param mu Vector of drift coefficients
     * @param sigma Vector of volatility coefficients
     * @param correlation Correlation matrix (n x n)
     */
    MultiAssetGBM(const std::vector<float>& mu, 
                  const std::vector<float>& sigma,
                  const std::vector<float>& correlation = {})
        : m_mu(mu)
        , m_sigma(sigma)
        , m_correlation(correlation)
        , m_dimension(mu.size())
    {
        if (m_correlation.empty()) {
            // Identity correlation matrix
            m_correlation.resize(m_dimension * m_dimension, 0.0f);
            for (int i = 0; i < m_dimension; i++) {
                m_correlation[i * m_dimension + i] = 1.0f;
            }
        }
        
        // Compute Cholesky decomposition for correlation matrix
        computeCholesky();
    }
    
    virtual std::vector<float> drift(const std::vector<float>& state, float t) const override {
        std::vector<float> result(m_dimension);
        for (int i = 0; i < m_dimension; i++) {
            result[i] = m_mu[i] * state[i];
        }
        return result;
    }
    
    virtual std::vector<float> diffusion(const std::vector<float>& state, float t) const override {
        // Return diagonal volatility matrix times state
        std::vector<float> result(m_dimension * m_dimension, 0.0f);
        for (int i = 0; i < m_dimension; i++) {
            result[i * m_dimension + i] = m_sigma[i] * state[i];
        }
        return result;
    }
    
    virtual int getDimension() const override { return m_dimension; }
    
    virtual std::string getName() const override { 
        return "Multi-Asset Geometric Brownian Motion"; 
    }
    
    virtual std::vector<float> getParameters() const override {
        std::vector<float> params;
        params.insert(params.end(), m_mu.begin(), m_mu.end());
        params.insert(params.end(), m_sigma.begin(), m_sigma.end());
        return params;
    }
    
    virtual void setParameters(const std::vector<float>& params) override {
        int n = m_dimension;
        if (params.size() >= n) {
            m_mu.assign(params.begin(), params.begin() + n);
        }
        if (params.size() >= 2 * n) {
            m_sigma.assign(params.begin() + n, params.begin() + 2 * n);
        }
    }
    
    // Get Cholesky decomposition of correlation matrix
    const std::vector<float>& getCholesky() const { return m_cholesky; }
    
    // Generate correlated normal random variables
    std::vector<float> generateCorrelatedNormals(const std::vector<float>& independent) const {
        return SdeUtils::matmul(m_cholesky, independent, m_dimension, m_dimension);
    }

private:
    std::vector<float> m_mu;
    std::vector<float> m_sigma;
    std::vector<float> m_correlation;
    std::vector<float> m_cholesky;
    int m_dimension;
    
    void computeCholesky() {
        // Simplified Cholesky decomposition
        int n = m_dimension;
        m_cholesky.resize(n * n, 0.0f);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = 0.0f;
                
                if (j == i) {
                    for (int k = 0; k < j; k++) {
                        sum += m_cholesky[j * n + k] * m_cholesky[j * n + k];
                    }
                    m_cholesky[j * n + j] = std::sqrt(m_correlation[j * n + j] - sum);
                } else {
                    for (int k = 0; k < j; k++) {
                        sum += m_cholesky[i * n + k] * m_cholesky[j * n + k];
                    }
                    m_cholesky[i * n + j] = (m_correlation[i * n + j] - sum) / 
                                            m_cholesky[j * n + j];
                }
            }
        }
    }
};

} // namespace sde
} // namespace mcgan