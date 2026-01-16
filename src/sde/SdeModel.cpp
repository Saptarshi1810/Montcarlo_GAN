#include "SdeModel.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mcgan {
namespace sde {

// ============================================================================
// CustomSde1D Implementation
// ============================================================================

std::vector<float> CustomSde1D::drift(const std::vector<float>& state, float t) const {
    if (state.empty()) {
        throw std::invalid_argument("State vector is empty");
    }
    return {m_driftFunc(state[0], t)};
}

std::vector<float> CustomSde1D::diffusion(const std::vector<float>& state, float t) const {
    if (state.empty()) {
        throw std::invalid_argument("State vector is empty");
    }
    return {m_diffusionFunc(state[0], t)};
}

// ============================================================================
// CustomSdeND Implementation
// ============================================================================

std::vector<float> CustomSdeND::drift(const std::vector<float>& state, float t) const {
    if (state.size() != static_cast<size_t>(m_dimension)) {
        throw std::invalid_argument("State dimension mismatch");
    }
    return m_driftFunc(state, t);
}

std::vector<float> CustomSdeND::diffusion(const std::vector<float>& state, float t) const {
    if (state.size() != static_cast<size_t>(m_dimension)) {
        throw std::invalid_argument("State dimension mismatch");
    }
    return m_diffusionFunc(state, t);
}

// ============================================================================
// SdeUtils Implementation
// ============================================================================

namespace SdeUtils {

std::vector<float> generateNormalVector(int dimension, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> result(dimension);
    
    for (int i = 0; i < dimension; i++) {
        result[i] = dist(rng);
    }
    
    return result;
}

float norm(const std::vector<float>& vec) {
    float sum = 0.0f;
    
    for (float v : vec) {
        sum += v * v;
    }
    
    return std::sqrt(sum);
}

std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for addition");
    }
    
    std::vector<float> result(a.size());
    
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
    
    return result;
}

std::vector<float> scale(const std::vector<float>& vec, float scalar) {
    std::vector<float> result(vec.size());
    
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = vec[i] * scalar;
    }
    
    return result;
}

std::vector<float> matmul(const std::vector<float>& matrix, 
                         const std::vector<float>& vec, 
                         int rows, int cols) {
    if (matrix.size() != static_cast<size_t>(rows * cols)) {
        throw std::invalid_argument("Matrix size mismatch");
    }
    
    if (vec.size() != static_cast<size_t>(cols)) {
        throw std::invalid_argument("Vector size mismatch for matrix multiplication");
    }
    
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