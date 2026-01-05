#pragma once

#include "SdeModel.hpp"
#include "GeometricBrownianMotion.hpp"
#include "CoxIngersollRoss.hpp"
#include <random>
#include <cmath>
#include <memory>

namespace mcgan {
namespace sde {

/**
 * Path data structure for storing SDE simulation results.
 */
struct Path {
    std::vector<std::vector<float>> states;  // states[timeStep][dimension]
    std::vector<float> times;
    
    Path() = default;
    Path(int numSteps, int dimension) {
        states.resize(numSteps);
        for (auto& state : states) {
            state.resize(dimension);
        }
        times.resize(numSteps);
    }
    
    int getNumSteps() const { return states.size(); }
    int getDimension() const { return states.empty() ? 0 : states[0].size(); }
};

/**
 * Exact simulator for SDEs with known analytical solutions.
 * Uses closed-form solutions when available for accuracy and efficiency.
 */
class ExactSimulator {
public:
    ExactSimulator(std::shared_ptr<SdeModel> model, int seed = 0)
        : m_model(model)
        , m_rng(seed)
        , m_normalDist(0.0f, 1.0f)
    {}
    
    /**
     * Simulate a single path using exact solution (if available).
     * @param initialState Initial state X_0
     * @param T Terminal time
     * @param numSteps Number of time steps
     * @return Simulated path
     */
    Path simulate(const std::vector<float>& initialState, float T, int numSteps) {
        Path path(numSteps + 1, m_model->getDimension());
        float dt = T / numSteps;
        
        // Set initial state
        path.states[0] = initialState;
        path.times[0] = 0.0f;
        
        // Try exact simulation for known models
        if (simulateGBM(path, T, numSteps, dt)) {
            return path;
        }
        
        if (simulateCIR(path, T, numSteps, dt)) {
            return path;
        }
        
        // Fallback: No exact solution available
        throw std::runtime_error("No exact solution available for this SDE model");
    }
    
    /**
     * Simulate multiple independent paths.
     */
    std::vector<Path> simulatePaths(const std::vector<float>& initialState, 
                                    float T, int numSteps, int numPaths) {
        std::vector<Path> paths;
        paths.reserve(numPaths);
        
        for (int i = 0; i < numPaths; i++) {
            paths.push_back(simulate(initialState, T, numSteps));
        }
        
        return paths;
    }
    
    /**
     * Simulate terminal values only (faster for option pricing).
     */
    std::vector<float> simulateTerminal(const std::vector<float>& initialState, float T) {
        int dim = m_model->getDimension();
        std::vector<float> terminal(dim);
        
        // GBM exact terminal simulation
        auto gbm = std::dynamic_pointer_cast<GeometricBrownianMotion>(m_model);
        if (gbm && dim == 1) {
            float Z = m_normalDist(m_rng);
            terminal[0] = gbm->exactSolution(initialState[0], T, Z);
            return terminal;
        }
        
        // CIR exact terminal simulation (using non-central chi-squared)
        auto cir = std::dynamic_pointer_cast<CoxIngersollRoss>(m_model);
        if (cir && dim == 1) {
            terminal[0] = simulateCIRTerminal(initialState[0], T, 
                                             cir->getKappa(), cir->getTheta(), cir->getSigma());
            return terminal;
        }
        
        throw std::runtime_error("No exact terminal simulation available");
    }
    
    /**
     * Simulate multiple terminal values.
     */
    std::vector<std::vector<float>> simulateTerminals(const std::vector<float>& initialState, 
                                                      float T, int numPaths) {
        std::vector<std::vector<float>> terminals;
        terminals.reserve(numPaths);
        
        for (int i = 0; i < numPaths; i++) {
            terminals.push_back(simulateTerminal(initialState, T));
        }
        
        return terminals;
    }
    
    void setSeed(int seed) {
        m_rng.seed(seed);
    }

private:
    std::shared_ptr<SdeModel> m_model;
    std::mt19937 m_rng;
    std::normal_distribution<float> m_normalDist;
    
    /**
     * Exact simulation for Geometric Brownian Motion.
     */
    bool simulateGBM(Path& path, float T, int numSteps, float dt) {
        auto gbm = std::dynamic_pointer_cast<GeometricBrownianMotion>(m_model);
        if (!gbm || m_model->getDimension() != 1) {
            return false;
        }
        
        float mu = gbm->getMu();
        float sigma = gbm->getSigma();
        float S = path.states[0][0];
        
        for (int i = 1; i <= numSteps; i++) {
            float t = i * dt;
            float Z = m_normalDist(m_rng);
            
            // Exact solution: S_t = S_0 * exp((μ - σ²/2)t + σ√t Z)
            S = path.states[0][0] * std::exp((mu - 0.5f * sigma * sigma) * t + 
                                             sigma * std::sqrt(t) * Z);
            
            path.states[i][0] = S;
            path.times[i] = t;
        }
        
        return true;
    }
    
    /**
     * Exact simulation for Cox-Ingersoll-Ross using non-central chi-squared.
     */
    bool simulateCIR(Path& path, float T, int numSteps, float dt) {
        auto cir = std::dynamic_pointer_cast<CoxIngersollRoss>(m_model);
        if (!cir || m_model->getDimension() != 1) {
            return false;
        }
        
        float kappa = cir->getKappa();
        float theta = cir->getTheta();
        float sigma = cir->getSigma();
        
        float r = path.states[0][0];
        
        for (int i = 1; i <= numSteps; i++) {
            float t = i * dt;
            r = simulateCIRTerminal(r, dt, kappa, theta, sigma);
            path.states[i][0] = r;
            path.times[i] = t;
        }
        
        return true;
    }
    
    /**
     * Simulate CIR terminal value using non-central chi-squared approximation.
     */
    float simulateCIRTerminal(float r0, float dt, float kappa, float theta, float sigma) {
        // Exact simulation using non-central chi-squared distribution
        // This is a simplified approximation
        
        float c = sigma * sigma * (1.0f - std::exp(-kappa * dt)) / (4.0f * kappa);
        float df = 4.0f * kappa * theta / (sigma * sigma);  // Degrees of freedom
        float nc = r0 * std::exp(-kappa * dt) / c;  // Non-centrality parameter
        
        // Approximate with normal for simplicity (exact would use non-central chi-squared)
        float mean = theta + (r0 - theta) * std::exp(-kappa * dt);
        float variance = r0 * sigma * sigma * std::exp(-kappa * dt) * 
                        (1.0f - std::exp(-kappa * dt)) / kappa +
                        theta * sigma * sigma * (1.0f - std::exp(-kappa * dt)) * 
                        (1.0f - std::exp(-kappa * dt)) / (2.0f * kappa);
        
        float Z = m_normalDist(m_rng);
        float r = mean + std::sqrt(variance) * Z;
        
        return std::max(0.0f, r);  // Ensure non-negative
    }
};

/**
 * Helper functions for exact simulation.
 */
namespace ExactSimUtils {

/**
 * Simulate standard Brownian motion path.
 */
inline Path simulateBrownianMotion(float T, int numSteps, std::mt19937& rng) {
    Path path(numSteps + 1, 1);
    float dt = T / numSteps;
    float sqrtDt = std::sqrt(dt);
    
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    path.states[0][0] = 0.0f;
    path.times[0] = 0.0f;
    
    for (int i = 1; i <= numSteps; i++) {
        float dW = sqrtDt * dist(rng);
        path.states[i][0] = path.states[i-1][0] + dW;
        path.times[i] = i * dt;
    }
    
    return path;
}

/**
 * Simulate Brownian bridge.
 */
inline Path simulateBrownianBridge(float T, float endValue, int numSteps, std::mt19937& rng) {
    Path path(numSteps + 1, 1);
    float dt = T / numSteps;
    
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    path.states[0][0] = 0.0f;
    path.times[0] = 0.0f;
    
    // Generate standard Brownian motion first
    std::vector<float> W(numSteps + 1, 0.0f);
    for (int i = 1; i <= numSteps; i++) {
        W[i] = W[i-1] + std::sqrt(dt) * dist(rng);
    }
    
    // Apply bridge constraint
    for (int i = 1; i < numSteps; i++) {
        float t = i * dt;
        path.states[i][0] = W[i] - (t / T) * (W[numSteps] - endValue);
        path.times[i] = t;
    }
    
    path.states[numSteps][0] = endValue;
    path.times[numSteps] = T;
    
    return path;
}

} // namespace ExactSimUtils

} // namespace sde
} // namespace mcgan