#pragma once

#include "SdeModel.hpp"
#include <random>
#include <memory>
#include <cmath>

namespace mcgan {
namespace sde {

// Forward declaration
struct Path;

/**
 * Base class for discrete-time SDE simulation schemes.
 */
class DiscreteScheme {
public:
    virtual ~DiscreteScheme() = default;
    
    /**
     * Take one step from current state.
     * @param state Current state
     * @param t Current time
     * @param dt Time step
     * @param dW Brownian increment (√dt * Z where Z ~ N(0,1))
     * @return Next state
     */
    virtual std::vector<float> step(const std::vector<float>& state, 
                                    float t, float dt, 
                                    const std::vector<float>& dW) = 0;
    
    /**
     * Simulate a complete path.
     */
    virtual Path simulate(const std::vector<float>& initialState, 
                         float T, int numSteps) = 0;
};

/**
 * Euler-Maruyama scheme (order 0.5 strong convergence).
 * 
 * X_{n+1} = X_n + μ(X_n, t_n)Δt + σ(X_n, t_n)ΔW_n
 * 
 * Simple and most commonly used method.
 */
class EulerMaruyama : public DiscreteScheme {
public:
    EulerMaruyama(std::shared_ptr<SdeModel> model, int seed = 0)
        : m_model(model)
        , m_rng(seed)
        , m_normalDist(0.0f, 1.0f)
    {}
    
    virtual std::vector<float> step(const std::vector<float>& state, 
                                    float t, float dt, 
                                    const std::vector<float>& dW) override {
        int dim = m_model->getDimension();
        std::vector<float> nextState(dim);
        
        auto drift = m_model->drift(state, t);
        auto diffusion = m_model->diffusion(state, t);
        
        for (int i = 0; i < dim; i++) {
            nextState[i] = state[i] + drift[i] * dt;
            
            // Add diffusion term
            if (diffusion.size() == dim) {
                // Diagonal diffusion
                nextState[i] += diffusion[i] * dW[i];
            } else {
                // Full diffusion matrix
                for (int j = 0; j < dim; j++) {
                    nextState[i] += diffusion[i * dim + j] * dW[j];
                }
            }
        }
        
        return nextState;
    }
    
    virtual Path simulate(const std::vector<float>& initialState, 
                         float T, int numSteps) override;

protected:
    std::shared_ptr<SdeModel> m_model;
    std::mt19937 m_rng;
    std::normal_distribution<float> m_normalDist;
};

/**
 * Milstein scheme (order 1.0 strong convergence).
 * 
 * Includes correction term for better accuracy:
 * X_{n+1} = X_n + μΔt + σΔW + 0.5σ(∂σ/∂x)(ΔW² - Δt)
 */
class Milstein : public DiscreteScheme {
public:
    Milstein(std::shared_ptr<SdeModel> model, int seed = 0)
        : m_model(model)
        , m_rng(seed)
        , m_normalDist(0.0f, 1.0f)
        , m_epsilon(1e-6f)
    {}
    
    virtual std::vector<float> step(const std::vector<float>& state, 
                                    float t, float dt, 
                                    const std::vector<float>& dW) override {
        int dim = m_model->getDimension();
        std::vector<float> nextState(dim);
        
        auto drift = m_model->drift(state, t);
        auto diffusion = m_model->diffusion(state, t);
        
        for (int i = 0; i < dim; i++) {
            nextState[i] = state[i] + drift[i] * dt;
            
            if (diffusion.size() == dim) {
                // Diagonal diffusion - compute derivative numerically
                std::vector<float> statePlus = state;
                statePlus[i] += m_epsilon;
                auto diffusionPlus = m_model->diffusion(statePlus, t);
                
                float dSigmaDx = (diffusionPlus[i] - diffusion[i]) / m_epsilon;
                
                // Milstein correction
                nextState[i] += diffusion[i] * dW[i] + 
                               0.5f * diffusion[i] * dSigmaDx * (dW[i] * dW[i] - dt);
            } else {
                // Full matrix - use Euler-Maruyama for simplicity
                for (int j = 0; j < dim; j++) {
                    nextState[i] += diffusion[i * dim + j] * dW[j];
                }
            }
        }
        
        return nextState;
    }
    
    virtual Path simulate(const std::vector<float>& initialState, 
                         float T, int numSteps) override;

protected:
    std::shared_ptr<SdeModel> m_model;
    std::mt19937 m_rng;
    std::normal_distribution<float> m_normalDist;
    float m_epsilon;
};

/**
 * Runge-Kutta scheme for SDEs (order 1.0 strong).
 * 
 * Stochastic Runge-Kutta method with improved stability.
 */
class RungeKutta : public DiscreteScheme {
public:
    RungeKutta(std::shared_ptr<SdeModel> model, int seed = 0)
        : m_model(model)
        , m_rng(seed)
        , m_normalDist(0.0f, 1.0f)
    {}
    
    virtual std::vector<float> step(const std::vector<float>& state, 
                                    float t, float dt, 
                                    const std::vector<float>& dW) override {
        int dim = m_model->getDimension();
        
        // K1
        auto drift1 = m_model->drift(state, t);
        auto diffusion1 = m_model->diffusion(state, t);
        
        // Intermediate state
        std::vector<float> stateInt(dim);
        for (int i = 0; i < dim; i++) {
            stateInt[i] = state[i] + drift1[i] * dt;
            if (diffusion1.size() == dim) {
                stateInt[i] += diffusion1[i] * std::sqrt(dt);
            }
        }
        
        // K2
        auto drift2 = m_model->drift(stateInt, t + dt);
        auto diffusion2 = m_model->diffusion(stateInt, t + dt);
        
        // Combine
        std::vector<float> nextState(dim);
        for (int i = 0; i < dim; i++) {
            nextState[i] = state[i] + 0.5f * (drift1[i] + drift2[i]) * dt;
            
            if (diffusion1.size() == dim) {
                nextState[i] += 0.5f * (diffusion1[i] + diffusion2[i]) * dW[i];
            }
        }
        
        return nextState;
    }
    
    virtual Path simulate(const std::vector<float>& initialState, 
                         float T, int numSteps) override;

protected:
    std::shared_ptr<SdeModel> m_model;
    std::mt19937 m_rng;
    std::normal_distribution<float> m_normalDist;
};

/**
 * Implicit Euler scheme for stiff SDEs.
 * 
 * X_{n+1} = X_n + μ(X_{n+1}, t_{n+1})Δt + σ(X_n, t_n)ΔW_n
 * 
 * Better stability for stiff problems (requires solving implicit equation).
 */
class ImplicitEuler : public DiscreteScheme {
public:
    ImplicitEuler(std::shared_ptr<SdeModel> model, int seed = 0, 
                  int maxIterations = 10, float tolerance = 1e-6f)
        : m_model(model)
        , m_rng(seed)
        , m_normalDist(0.0f, 1.0f)
        , m_maxIterations(maxIterations)
        , m_tolerance(tolerance)
    {}
    
    virtual std::vector<float> step(const std::vector<float>& state, 
                                    float t, float dt, 
                                    const std::vector<float>& dW) override {
        int dim = m_model->getDimension();
        
        // Get diffusion (explicit)
        auto diffusion = m_model->diffusion(state, t);
        
        // Initial guess (explicit Euler)
        auto drift = m_model->drift(state, t);
        std::vector<float> nextState(dim);
        for (int i = 0; i < dim; i++) {
            nextState[i] = state[i] + drift[i] * dt;
            if (diffusion.size() == dim) {
                nextState[i] += diffusion[i] * dW[i];
            }
        }
        
        // Fixed-point iteration to solve implicit equation
        for (int iter = 0; iter < m_maxIterations; iter++) {
            auto driftNext = m_model->drift(nextState, t + dt);
            
            std::vector<float> newState(dim);
            float maxDiff = 0.0f;
            
            for (int i = 0; i < dim; i++) {
                newState[i] = state[i] + driftNext[i] * dt;
                if (diffusion.size() == dim) {
                    newState[i] += diffusion[i] * dW[i];
                }
                
                float diff = std::abs(newState[i] - nextState[i]);
                maxDiff = std::max(maxDiff, diff);
            }
            
            nextState = newState;
            
            if (maxDiff < m_tolerance) {
                break;
            }
        }
        
        return nextState;
    }
    
    virtual Path simulate(const std::vector<float>& initialState, 
                         float T, int numSteps) override;

protected:
    std::shared_ptr<SdeModel> m_model;
    std::mt19937 m_rng;
    std::normal_distribution<float> m_normalDist;
    int m_maxIterations;
    float m_tolerance;
};

// Implementations of simulate methods
inline Path EulerMaruyama::simulate(const std::vector<float>& initialState, 
                                   float T, int numSteps) {
    int dim = m_model->getDimension();
    Path path(numSteps + 1, dim);
    
    float dt = T / numSteps;
    float sqrtDt = std::sqrt(dt);
    
    path.states[0] = initialState;
    path.times[0] = 0.0f;
    
    for (int i = 1; i <= numSteps; i++) {
        // Generate Brownian increments
        std::vector<float> dW = SdeUtils::generateNormalVector(dim, m_rng);
        for (float& w : dW) {
            w *= sqrtDt;
        }
        
        float t = (i - 1) * dt;
        path.states[i] = step(path.states[i-1], t, dt, dW);
        path.times[i] = i * dt;
    }
    
    return path;
}

inline Path Milstein::simulate(const std::vector<float>& initialState, 
                               float T, int numSteps) {
    int dim = m_model->getDimension();
    Path path(numSteps + 1, dim);
    
    float dt = T / numSteps;
    float sqrtDt = std::sqrt(dt);
    
    path.states[0] = initialState;
    path.times[0] = 0.0f;
    
    for (int i = 1; i <= numSteps; i++) {
        std::vector<float> dW = SdeUtils::generateNormalVector(dim, m_rng);
        for (float& w : dW) {
            w *= sqrtDt;
        }
        
        float t = (i - 1) * dt;
        path.states[i] = step(path.states[i-1], t, dt, dW);
        path.times[i] = i * dt;
    }
    
    return path;
}

inline Path RungeKutta::simulate(const std::vector<float>& initialState, 
                                float T, int numSteps) {
    int dim = m_model->getDimension();
    Path path(numSteps + 1, dim);
    
    float dt = T / numSteps;
    float sqrtDt = std::sqrt(dt);
    
    path.states[0] = initialState;
    path.times[0] = 0.0f;
    
    for (int i = 1; i <= numSteps; i++) {
        std::vector<float> dW = SdeUtils::generateNormalVector(dim, m_rng);
        for (float& w : dW) {
            w *= sqrtDt;
        }
        
        float t = (i - 1) * dt;
        path.states[i] = step(path.states[i-1], t, dt, dW);
        path.times[i] = i * dt;
    }
    
    return path;
}

inline Path ImplicitEuler::simulate(const std::vector<float>& initialState, 
                                   float T, int numSteps) {
    int dim = m_model->getDimension();
    Path path(numSteps + 1, dim);
    
    float dt = T / numSteps;
    float sqrtDt = std::sqrt(dt);
    
    path.states[0] = initialState;
    path.times[0] = 0.0f;
    
    for (int i = 1; i <= numSteps; i++) {
        std::vector<float> dW = SdeUtils::generateNormalVector(dim, m_rng);
        for (float& w : dW) {
            w *= sqrtDt;
        }
        
        float t = (i - 1) * dt;
        path.states[i] = step(path.states[i-1], t, dt, dW);
        path.times[i] = i * dt;
    }
    
    return path;
}

} // namespace sde
} // namespace mcgan