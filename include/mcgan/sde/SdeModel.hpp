#pragma once

#include "../core/Types.hpp"
#include <memory>
#include <vector>
#include <functional>

namespace mcgan {
namespace sde {

/**
 * Abstract base class for Stochastic Differential Equation models.
 * Represents dX_t = drift(X_t, t)dt + diffusion(X_t, t)dW_t
 */
class SdeModel {
public:
    virtual ~SdeModel() = default;
    
    /**
     * Compute the drift coefficient μ(X_t, t).
     * @param state Current state X_t
     * @param t Current time
     * @return Drift vector
     */
    virtual std::vector<float> drift(const std::vector<float>& state, float t) const = 0;
    
    /**
     * Compute the diffusion coefficient σ(X_t, t).
     * @param state Current state X_t
     * @param t Current time
     * @return Diffusion matrix (flattened)
     */
    virtual std::vector<float> diffusion(const std::vector<float>& state, float t) const = 0;
    
    /**
     * Get the dimension of the state space.
     */
    virtual int getDimension() const = 0;
    
    /**
     * Get the name of the SDE model.
     */
    virtual std::string getName() const = 0;
    
    /**
     * Check if the diffusion is constant (simplifies simulation).
     */
    virtual bool hasConstantDiffusion() const { return false; }
    
    /**
     * Get parameters of the model.
     */
    virtual std::vector<float> getParameters() const = 0;
    
    /**
     * Set parameters of the model.
     */
    virtual void setParameters(const std::vector<float>& params) = 0;
};

/**
 * Generic 1D SDE model with custom drift and diffusion functions.
 */
class CustomSde1D : public SdeModel {
public:
    using DriftFunc = std::function<float(float, float)>;
    using DiffusionFunc = std::function<float(float, float)>;
    
    CustomSde1D(DriftFunc drift, DiffusionFunc diffusion, const std::string& name = "Custom1D")
        : m_driftFunc(drift)
        , m_diffusionFunc(diffusion)
        , m_name(name)
    {}
    
    virtual std::vector<float> drift(const std::vector<float>& state, float t) const override {
        return {m_driftFunc(state[0], t)};
    }
    
    virtual std::vector<float> diffusion(const std::vector<float>& state, float t) const override {
        return {m_diffusionFunc(state[0], t)};
    }
    
    virtual int getDimension() const override { return 1; }
    virtual std::string getName() const override { return m_name; }
    virtual std::vector<float> getParameters() const override { return {}; }
    virtual void setParameters(const std::vector<float>& params) override {}

private:
    DriftFunc m_driftFunc;
    DiffusionFunc m_diffusionFunc;
    std::string m_name;
};

/**
 * Multi-dimensional SDE model with custom drift and diffusion.
 */
class CustomSdeND : public SdeModel {
public:
    using DriftFunc = std::function<std::vector<float>(const std::vector<float>&, float)>;
    using DiffusionFunc = std::function<std::vector<float>(const std::vector<float>&, float)>;
    
    CustomSdeND(int dimension, DriftFunc drift, DiffusionFunc diffusion, 
                const std::string& name = "CustomND")
        : m_dimension(dimension)
        , m_driftFunc(drift)
        , m_diffusionFunc(diffusion)
        , m_name(name)
    {}
    
    virtual std::vector<float> drift(const std::vector<float>& state, float t) const override {
        return m_driftFunc(state, t);
    }
    
    virtual std::vector<float> diffusion(const std::vector<float>& state, float t) const override {
        return m_diffusionFunc(state, t);
    }
    
    virtual int getDimension() const override { return m_dimension; }
    virtual std::string getName() const override { return m_name; }
    virtual std::vector<float> getParameters() const override { return {}; }
    virtual void setParameters(const std::vector<float>& params) override {}

private:
    int m_dimension;
    DriftFunc m_driftFunc;
    DiffusionFunc m_diffusionFunc;
    std::string m_name;
};

/**
 * Helper functions for common SDE operations.
 */
namespace SdeUtils {

/**
 * Generate standard normal random variables.
 */
inline std::vector<float> generateNormalVector(int dimension, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> result(dimension);
    for (int i = 0; i < dimension; i++) {
        result[i] = dist(rng);
    }
    return result;
}

/**
 * Compute Euclidean norm of a vector.
 */
inline float norm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float v : vec) {
        sum += v * v;
    }
    return std::sqrt(sum);
}

/**
 * Vector addition.
 */
inline std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * Scalar multiplication.
 */
inline std::vector<float> scale(const std::vector<float>& vec, float scalar) {
    std::vector<float> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = vec[i] * scalar;
    }
    return result;
}

/**
 * Matrix-vector multiplication (for diffusion matrix).
 */
inline std::vector<float> matmul(const std::vector<float>& matrix, 
                                const std::vector<float>& vec, 
                                int rows, int cols) {
    std::vector<float> result(rows, 0.0f);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vec[j];
        }
    }
    return result;
}

} // namespace SdeUtils

} // namespace sde
} // namespace mcgan