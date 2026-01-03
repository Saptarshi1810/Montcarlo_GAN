#pragma once

#include "SdeModel.hpp"
#include <cmath>
#include <algorithm>

namespace mcgan {
namespace sde {

/**
 * Cox-Ingersoll-Ross (CIR) model.
 * 
 * Used for modeling interest rates and volatility.
 * dr_t = κ(θ - r_t)dt + σ√r_t dW_t
 * 
 * where:
 *   r_t = short rate (interest rate or volatility) at time t
 *   κ = speed of mean reversion
 *   θ = long-term mean level
 *   σ = volatility
 *   
 * The square-root diffusion ensures r_t remains non-negative
 * under the Feller condition: 2κθ ≥ σ²
 */
class CoxIngersollRoss : public SdeModel {
public:
    /**
     * Constructor.
     * @param kappa Speed of mean reversion
     * @param theta Long-term mean level
     * @param sigma Volatility coefficient
     */
    CoxIngersollRoss(float kappa = 0.5f, float theta = 0.05f, float sigma = 0.1f)
        : m_kappa(kappa)
        , m_theta(theta)
        , m_sigma(sigma)
    {}
    
    virtual std::vector<float> drift(const std::vector<float>& state, float t) const override {
        // κ(θ - r_t)
        float r = std::max(0.0f, state[0]);  // Ensure non-negative
        return {m_kappa * (m_theta - r)};
    }
    
    virtual std::vector<float> diffusion(const std::vector<float>& state, float t) const override {
        // σ√r_t
        float r = std::max(0.0f, state[0]);  // Ensure non-negative
        return {m_sigma * std::sqrt(r)};
    }
    
    virtual int getDimension() const override { return 1; }
    
    virtual std::string getName() const override { 
        return "Cox-Ingersoll-Ross"; 
    }
    
    virtual bool hasConstantDiffusion() const override { return false; }
    
    virtual std::vector<float> getParameters() const override {
        return {m_kappa, m_theta, m_sigma};
    }
    
    virtual void setParameters(const std::vector<float>& params) override {
        if (params.size() >= 1) m_kappa = params[0];
        if (params.size() >= 2) m_theta = params[1];
        if (params.size() >= 3) m_sigma = params[2];
    }
    
    /**
     * Check if Feller condition is satisfied.
     * If true, the process stays strictly positive.
     */
    bool satisfiesFellerCondition() const {
        return 2.0f * m_kappa * m_theta >= m_sigma * m_sigma;
    }
    
    /**
     * Compute the long-term variance.
     */
    float getLongTermVariance() const {
        return m_sigma * m_sigma * m_theta / (2.0f * m_kappa);
    }
    
    /**
     * Compute expected value at time t given initial value r0.
     */
    float expectedValue(float r0, float t) const {
        return r0 * std::exp(-m_kappa * t) + m_theta * (1.0f - std::exp(-m_kappa * t));
    }
    
    /**
     * Compute variance at time t given initial value r0.
     */
    float variance(float r0, float t) const {
        float expTerm = std::exp(-m_kappa * t);
        return (r0 * m_sigma * m_sigma / m_kappa * (expTerm - expTerm * expTerm) +
                m_theta * m_sigma * m_sigma / (2.0f * m_kappa) * (1.0f - expTerm) * (1.0f - expTerm));
    }
    
    // Getters
    float getKappa() const { return m_kappa; }
    float getTheta() const { return m_theta; }
    float getSigma() const { return m_sigma; }
    
    // Setters
    void setKappa(float kappa) { m_kappa = kappa; }
    void setTheta(float theta) { m_theta = theta; }
    void setSigma(float sigma) { m_sigma = sigma; }

private:
    float m_kappa;  // Mean reversion speed
    float m_theta;  // Long-term mean
    float m_sigma;  // Volatility
};

/**
 * Extended CIR model with time-dependent parameters.
 */
class CIRTimeDep : public SdeModel {
public:
    using ParamFunc = std::function<float(float)>;
    
    CIRTimeDep(ParamFunc kappa, ParamFunc theta, ParamFunc sigma)
        : m_kappaFunc(kappa)
        , m_thetaFunc(theta)
        , m_sigmaFunc(sigma)
    {}
    
    virtual std::vector<float> drift(const std::vector<float>& state, float t) const override {
        float r = std::max(0.0f, state[0]);
        float kappa = m_kappaFunc(t);
        float theta = m_thetaFunc(t);
        return {kappa * (theta - r)};
    }
    
    virtual std::vector<float> diffusion(const std::vector<float>& state, float t) const override {
        float r = std::max(0.0f, state[0]);
        float sigma = m_sigmaFunc(t);
        return {sigma * std::sqrt(r)};
    }
    
    virtual int getDimension() const override { return 1; }
    
    virtual std::string getName() const override { 
        return "Cox-Ingersoll-Ross (Time-Dependent)"; 
    }
    
    virtual std::vector<float> getParameters() const override { return {}; }
    virtual void setParameters(const std::vector<float>& params) override {}

private:
    ParamFunc m_kappaFunc;
    ParamFunc m_thetaFunc;
    ParamFunc m_sigmaFunc;
};

/**
 * Multi-factor CIR model.
 * Used for modeling multiple correlated rates or volatilities.
 */
class MultiFactorCIR : public SdeModel {
public:
    struct Factor {
        float kappa;
        float theta;
        float sigma;
        
        Factor(float k = 0.5f, float t = 0.05f, float s = 0.1f)
            : kappa(k), theta(t), sigma(s) {}
    };
    
    MultiFactorCIR(const std::vector<Factor>& factors)
        : m_factors(factors)
        , m_dimension(factors.size())
    {}
    
    virtual std::vector<float> drift(const std::vector<float>& state, float t) const override {
        std::vector<float> result(m_dimension);
        for (int i = 0; i < m_dimension; i++) {
            float r = std::max(0.0f, state[i]);
            result[i] = m_factors[i].kappa * (m_factors[i].theta - r);
        }
        return result;
    }
    
    virtual std::vector<float> diffusion(const std::vector<float>& state, float t) const override {
        // Diagonal diffusion matrix
        std::vector<float> result(m_dimension * m_dimension, 0.0f);
        for (int i = 0; i < m_dimension; i++) {
            float r = std::max(0.0f, state[i]);
            result[i * m_dimension + i] = m_factors[i].sigma * std::sqrt(r);
        }
        return result;
    }
    
    virtual int getDimension() const override { return m_dimension; }
    
    virtual std::string getName() const override { 
        return "Multi-Factor Cox-Ingersoll-Ross"; 
    }
    
    virtual std::vector<float> getParameters() const override {
        std::vector<float> params;
        for (const auto& factor : m_factors) {
            params.push_back(factor.kappa);
            params.push_back(factor.theta);
            params.push_back(factor.sigma);
        }
        return params;
    }
    
    virtual void setParameters(const std::vector<float>& params) override {
        int idx = 0;
        for (size_t i = 0; i < m_factors.size() && idx + 2 < params.size(); i++) {
            m_factors[i].kappa = params[idx++];
            m_factors[i].theta = params[idx++];
            m_factors[i].sigma = params[idx++];
        }
    }
    
    const std::vector<Factor>& getFactors() const { return m_factors; }

private:
    std::vector<Factor> m_factors;
    int m_dimension;
};

} // namespace sde
} // namespace mcgan