#include "DiscreteScheme.hpp"
#include "ExactSimulator.hpp"
#include <cmath>
#include <algorithm>

namespace mcgan {
namespace sde {

// ============================================================================
// EulerMaruyama Implementation
// ============================================================================

std::vector<float> EulerMaruyama::step(
    const std::vector<float>& state, float t, float dt, 
    const std::vector<float>& dW) {
    
    int dim = m_model->getDimension();
    std::vector<float> nextState(dim);
    
    auto drift = m_model->drift(state, t);
    auto diffusion = m_model->diffusion(state, t);
    
    for (int i = 0; i < dim; i++) {
        // Drift term
        nextState[i] = state[i] + drift[i] * dt;
        
        // Diffusion term
        if (diffusion.size() == static_cast<size_t>(dim)) {
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

Path EulerMaruyama::simulate(const std::vector<float>& initialState, 
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

// ============================================================================
// Milstein Implementation
// ============================================================================

std::vector<float> Milstein::step(
    const std::vector<float>& state, float t, float dt, 
    const std::vector<float>& dW) {
    
    int dim = m_model->getDimension();
    std::vector<float> nextState(dim);
    
    auto drift = m_model->drift(state, t);
    auto diffusion = m_model->diffusion(state, t);
    
    for (int i = 0; i < dim; i++) {
        // Drift term
        nextState[i] = state[i] + drift[i] * dt;
        
        if (diffusion.size() == static_cast<size_t>(dim)) {
            // Diagonal diffusion with Milstein correction
            
            // Compute numerical derivative of diffusion
            std::vector<float> statePlus = state;
            statePlus[i] += m_epsilon;
            auto diffusionPlus = m_model->diffusion(statePlus, t);
            
            float dSigmaDx = (diffusionPlus[i] - diffusion[i]) / m_epsilon;
            
            // Euler-Maruyama term
            nextState[i] += diffusion[i] * dW[i];
            
            // Milstein correction: 0.5 * σ * (∂σ/∂x) * (ΔW² - Δt)
            nextState[i] += 0.5f * diffusion[i] * dSigmaDx * (dW[i] * dW[i] - dt);
        } else {
            // Full matrix - use Euler-Maruyama for simplicity
            for (int j = 0; j < dim; j++) {
                nextState[i] += diffusion[i * dim + j] * dW[j];
            }
        }
    }
    
    return nextState;
}

Path Milstein::simulate(const std::vector<float>& initialState, 
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

// ============================================================================
// RungeKutta Implementation
// ============================================================================

std::vector<float> RungeKutta::step(
    const std::vector<float>& state, float t, float dt, 
    const std::vector<float>& dW) {
    
    int dim = m_model->getDimension();
    
    // Stage 1
    auto drift1 = m_model->drift(state, t);
    auto diffusion1 = m_model->diffusion(state, t);
    
    // Intermediate state for stage 2
    std::vector<float> stateInt(dim);
    for (int i = 0; i < dim; i++) {
        stateInt[i] = state[i] + drift1[i] * dt;
        
        if (diffusion1.size() == static_cast<size_t>(dim)) {
            // For RK, use sqrt(dt) * standard normal
            stateInt[i] += diffusion1[i] * std::sqrt(dt);
        }
    }
    
    // Stage 2
    auto drift2 = m_model->drift(stateInt, t + dt);
    auto diffusion2 = m_model->diffusion(stateInt, t + dt);
    
    // Combine stages
    std::vector<float> nextState(dim);
    for (int i = 0; i < dim; i++) {
        nextState[i] = state[i] + 0.5f * (drift1[i] + drift2[i]) * dt;
        
        if (diffusion1.size() == static_cast<size_t>(dim)) {
            // Average diffusion coefficients
            nextState[i] += 0.5f * (diffusion1[i] + diffusion2[i]) * dW[i];
        } else {
            // Full matrix case
            for (int j = 0; j < dim; j++) {
                nextState[i] += 0.5f * (diffusion1[i * dim + j] + 
                                       diffusion2[i * dim + j]) * dW[j];
            }
        }
    }
    
    return nextState;
}

Path RungeKutta::simulate(const std::vector<float>& initialState, 
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

// ============================================================================
// ImplicitEuler Implementation
// ============================================================================

std::vector<float> ImplicitEuler::step(
    const std::vector<float>& state, float t, float dt, 
    const std::vector<float>& dW) {
    
    int dim = m_model->getDimension();
    
    // Get explicit diffusion term
    auto diffusion = m_model->diffusion(state, t);
    
    // Initial guess using explicit Euler
    auto drift = m_model->drift(state, t);
    std::vector<float> nextState(dim);
    
    for (int i = 0; i < dim; i++) {
        nextState[i] = state[i] + drift[i] * dt;
        
        if (diffusion.size() == static_cast<size_t>(dim)) {
            nextState[i] += diffusion[i] * dW[i];
        } else {
            for (int j = 0; j < dim; j++) {
                nextState[i] += diffusion[i * dim + j] * dW[j];
            }
        }
    }
    
    // Fixed-point iteration to solve implicit equation
    // X_{n+1} = X_n + μ(X_{n+1}, t_{n+1})Δt + σ(X_n, t_n)ΔW
    for (int iter = 0; iter < m_maxIterations; iter++) {
        auto driftNext = m_model->drift(nextState, t + dt);
        
        std::vector<float> newState(dim);
        float maxDiff = 0.0f;
        
        for (int i = 0; i < dim; i++) {
            newState[i] = state[i] + driftNext[i] * dt;
            
            if (diffusion.size() == static_cast<size_t>(dim)) {
                newState[i] += diffusion[i] * dW[i];
            } else {
                for (int j = 0; j < dim; j++) {
                    newState[i] += diffusion[i * dim + j] * dW[j];
                }
            }
            
            float diff = std::abs(newState[i] - nextState[i]);
            maxDiff = std::max(maxDiff, diff);
        }
        
        nextState = newState;
        
        // Check convergence
        if (maxDiff < m_tolerance) {
            break;
        }
    }
    
    return nextState;
}

Path ImplicitEuler::simulate(const std::vector<float>& initialState, 
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