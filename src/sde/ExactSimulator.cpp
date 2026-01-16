#include "ExactSimulator.hpp"
#include "GeometricBrownianMotion.hpp"
#include "CoxIngersollRoss.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace mcgan {
namespace sde {

// ============================================================================
// Path Implementation
// ============================================================================

int Path::getNumSteps() const {
    return static_cast<int>(states.size());
}

int Path::getDimension() const {
    return states.empty() ? 0 : static_cast<int>(states[0].size());
}

// ============================================================================
// ExactSimulator Implementation
// ============================================================================

Path ExactSimulator::simulate(const std::vector<float>& initialState, 
                              float T, int numSteps) {
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
    throw std::runtime_error("No exact solution available for this SDE model: " + 
                           m_model->getName());
}

std::vector<Path> ExactSimulator::simulatePaths(
    const std::vector<float>& initialState, float T, int numSteps, int numPaths) {
    
    std::vector<Path> paths;
    paths.reserve(numPaths);
    
    for (int i = 0; i < numPaths; i++) {
        paths.push_back(simulate(initialState, T, numSteps));
    }
    
    return paths;
}

std::vector<float> ExactSimulator::simulateTerminal(
    const std::vector<float>& initialState, float T) {
    
    int dim = m_model->getDimension();
    std::vector<float> terminal(dim);
    
    // Try GBM exact terminal simulation
    auto gbm = std::dynamic_pointer_cast<GeometricBrownianMotion>(m_model);
    if (gbm && dim == 1) {
        float Z = m_normalDist(m_rng);
        terminal[0] = gbm->exactSolution(initialState[0], T, Z);
        return terminal;
    }
    
    // Try CIR exact terminal simulation
    auto cir = std::dynamic_pointer_cast<CoxIngersollRoss>(m_model);
    if (cir && dim == 1) {
        terminal[0] = simulateCIRTerminal(initialState[0], T, 
                                         cir->getKappa(), cir->getTheta(), cir->getSigma());
        return terminal;
    }
    
    throw std::runtime_error("No exact terminal simulation available for: " + 
                           m_model->getName());
}

std::vector<std::vector<float>> ExactSimulator::simulateTerminals(
    const std::vector<float>& initialState, float T, int numPaths) {
    
    std::vector<std::vector<float>> terminals;
    terminals.reserve(numPaths);
    
    for (int i = 0; i < numPaths; i++) {
        terminals.push_back(simulateTerminal(initialState, T));
    }
    
    return terminals;
}

void ExactSimulator::setSeed(int seed) {
    m_rng.seed(seed);
}

// ============================================================================
// Private Helper Methods
// ============================================================================

bool ExactSimulator::simulateGBM(Path& path, float T, int numSteps, float dt) {
    auto gbm = std::dynamic_pointer_cast<GeometricBrownianMotion>(m_model);
    if (!gbm || m_model->getDimension() != 1) {
        return false;
    }
    
    float mu = gbm->getMu();
    float sigma = gbm->getSigma();
    
    // Simulate using exact solution at each time step
    for (int i = 1; i <= numSteps; i++) {
        float t = i * dt;
        float Z = m_normalDist(m_rng);
        
        // Exact solution: S_t = S_0 * exp((μ - σ²/2)t + σ√t Z)
        float drift = (mu - 0.5f * sigma * sigma) * t;
        float diffusion = sigma * std::sqrt(t) * Z;
        
        path.states[i][0] = path.states[0][0] * std::exp(drift + diffusion);
        path.times[i] = t;
    }
    
    return true;
}

bool ExactSimulator::simulateCIR(Path& path, float T, int numSteps, float dt) {
    auto cir = std::dynamic_pointer_cast<CoxIngersollRoss>(m_model);
    if (!cir || m_model->getDimension() != 1) {
        return false;
    }
    
    float kappa = cir->getKappa();
    float theta = cir->getTheta();
    float sigma = cir->getSigma();
    
    float r = path.states[0][0];
    
    // Simulate step by step using exact transition distribution
    for (int i = 1; i <= numSteps; i++) {
        r = simulateCIRTerminal(r, dt, kappa, theta, sigma);
        path.states[i][0] = r;
        path.times[i] = i * dt;
    }
    
    return true;
}

float ExactSimulator::simulateCIRTerminal(float r0, float dt, 
                                         float kappa, float theta, float sigma) {
    // Exact simulation using non-central chi-squared distribution
    // This implementation uses a Gaussian approximation for simplicity
    // A full implementation would use the exact non-central chi-squared
    
    // Compute conditional moments
    float expKappaDt = std::exp(-kappa * dt);
    float mean = r0 * expKappaDt + theta * (1.0f - expKappaDt);
    
    // Variance formula for CIR process
    float var1 = r0 * sigma * sigma * expKappaDt * (1.0f - expKappaDt) / kappa;
    float var2 = theta * sigma * sigma * (1.0f - expKappaDt) * (1.0f - expKappaDt) / (2.0f * kappa);
    float variance = var1 + var2;
    
    // Generate from approximate normal distribution
    float Z = m_normalDist(m_rng);
    float r = mean + std::sqrt(std::max(0.0f, variance)) * Z;
    
    // Ensure non-negativity (important for CIR)
    return std::max(0.0f, r);
}

// ============================================================================
// ExactSimUtils Implementation
// ============================================================================

namespace ExactSimUtils {

Path simulateBrownianMotion(float T, int numSteps, std::mt19937& rng) {
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

Path simulateBrownianBridge(float T, float endValue, int numSteps, std::mt19937& rng) {
    Path path(numSteps + 1, 1);
    float dt = T / numSteps;
    
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    path.states[0][0] = 0.0f;
    path.times[0] = 0.0f;
    
    // First generate standard Brownian motion
    std::vector<float> W(numSteps + 1, 0.0f);
    float sqrtDt = std::sqrt(dt);
    
    for (int i = 1; i <= numSteps; i++) {
        W[i] = W[i-1] + sqrtDt * dist(rng);
    }
    
    // Apply Brownian bridge constraint: B_t = W_t - (t/T)(W_T - b)
    for (int i = 1; i < numSteps; i++) {
        float t = i * dt;
        path.states[i][0] = W[i] - (t / T) * (W[numSteps] - endValue);
        path.times[i] = t;
    }
    
    // Set endpoint exactly
    path.states[numSteps][0] = endValue;
    path.times[numSteps] = T;
    
    return path;
}

} // namespace ExactSimUtils

} // namespace sde
} // namespace mcgan