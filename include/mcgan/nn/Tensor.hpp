#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace mcgan {
namespace nn {

/**
 * Multi-dimensional tensor for neural network operations.
 * Supports various operations needed for deep learning.
 */
class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, float fillValue);
    Tensor(const std::vector<float>& data, const std::vector<int>& shape);
    
    // Copy and move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Shape information
    const std::vector<int>& shape() const { return m_shape; }
    int ndim() const { return m_shape.size(); }
    int size() const { return m_size; }
    int size(int dim) const;
    
    // Data access
    float* data() { return m_data.data(); }
    const float* data() const { return m_data.data(); }
    std::vector<float>& dataVec() { return m_data; }
    const std::vector<float>& dataVec() const { return m_data; }
    
    // Element access
    float& operator[](int idx) { return m_data[idx]; }
    const float& operator[](int idx) const { return m_data[idx]; }
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;
    
    // Reshaping
    Tensor reshape(const std::vector<int>& newShape) const;
    Tensor flatten() const;
    Tensor squeeze() const;
    Tensor unsqueeze(int dim) const;
    
    // Arithmetic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;
    
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    Tensor& operator+=(float scalar);
    Tensor& operator-=(float scalar);
    Tensor& operator*=(float scalar);
    Tensor& operator/=(float scalar);
    
    // Matrix operations
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    Tensor transpose(int dim0, int dim1) const;
    
    // Activation functions
    Tensor relu() const;
    Tensor leakyRelu(float alpha = 0.01f) const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor softmax(int dim = -1) const;
    
    // Gradient operations
    Tensor reluGradient() const;
    Tensor leakyReluGradient(float alpha = 0.01f) const;
    Tensor sigmoidGradient() const;
    Tensor tanhGradient() const;
    
    // Reduction operations
    float sum() const;
    float mean() const;
    float max() const;
    float min() const;
    Tensor sum(int dim, bool keepDim = false) const;
    Tensor mean(int dim, bool keepDim = false) const;
    
    // Other operations
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor pow(float exponent) const;
    Tensor clip(float minVal, float maxVal) const;
    
    // Initialization
    void fill(float value);
    void zero();
    void ones();
    void randomUniform(float min = 0.0f, float max = 1.0f, int seed = 0);
    void randomNormal(float mean = 0.0f, float std = 1.0f, int seed = 0);
    void xavier(int fanIn, int fanOut, int seed = 0);
    void he(int fanIn, int seed = 0);
    
    // Utility
    bool isCompatible(const Tensor& other) const;
    void print() const;

private:
    std::vector<float> m_data;
    std::vector<int> m_shape;
    std::vector<int> m_strides;
    int m_size;
    
    void computeStrides();
    int computeIndex(const std::vector<int>& indices) const;
    static std::vector<int> inferShape(int totalSize, const std::vector<int>& shape);
};

// Implementation

inline Tensor::Tensor() 
    : m_shape({0})
    , m_size(0) 
{
    computeStrides();
}

inline Tensor::Tensor(const std::vector<int>& shape) 
    : m_shape(shape)
{
    m_size = 1;
    for (int dim : m_shape) {
        m_size *= dim;
    }
    m_data.resize(m_size, 0.0f);
    computeStrides();
}

inline Tensor::Tensor(const std::vector<int>& shape, float fillValue) 
    : m_shape(shape)
{
    m_size = 1;
    for (int dim : m_shape) {
        m_size *= dim;
    }
    m_data.resize(m_size, fillValue);
    computeStrides();
}

inline Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape) 
    : m_data(data)
    , m_shape(shape)
{
    m_size = 1;
    for (int dim : m_shape) {
        m_size *= dim;
    }
    if (m_size != static_cast<int>(data.size())) {
        throw std::runtime_error("Data size doesn't match shape");
    }
    computeStrides();
}

inline Tensor::Tensor(const Tensor& other)
    : m_data(other.m_data)
    , m_shape(other.m_shape)
    , m_strides(other.m_strides)
    , m_size(other.m_size)
{
}

inline Tensor::Tensor(Tensor&& other) noexcept
    : m_data(std::move(other.m_data))
    , m_shape(std::move(other.m_shape))
    , m_strides(std::move(other.m_strides))
    , m_size(other.m_size)
{
}

inline Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        m_data = other.m_data;
        m_shape = other.m_shape;
        m_strides = other.m_strides;
        m_size = other.m_size;
    }
    return *this;
}

inline Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        m_data = std::move(other.m_data);
        m_shape = std::move(other.m_shape);
        m_strides = std::move(other.m_strides);
        m_size = other.m_size;
    }
    return *this;
}

inline int Tensor::size(int dim) const {
    if (dim < 0) dim += m_shape.size();
    if (dim < 0 || dim >= static_cast<int>(m_shape.size())) {
        throw std::out_of_range("Dimension out of range");
    }
    return m_shape[dim];
}

inline float& Tensor::at(const std::vector<int>& indices) {
    return m_data[computeIndex(indices)];
}

inline const float& Tensor::at(const std::vector<int>& indices) const {
    return m_data[computeIndex(indices)];
}

inline void Tensor::computeStrides() {
    m_strides.resize(m_shape.size());
    int stride = 1;
    for (int i = m_shape.size() - 1; i >= 0; i--) {
        m_strides[i] = stride;
        stride *= m_shape[i];
    }
}

inline int Tensor::computeIndex(const std::vector<int>& indices) const {
    if (indices.size() != m_shape.size()) {
        throw std::runtime_error("Number of indices doesn't match tensor dimensions");
    }
    
    int idx = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        idx += indices[i] * m_strides[i];
    }
    return idx;
}

inline Tensor Tensor::reshape(const std::vector<int>& newShape) const {
    int newSize = 1;
    int inferDim = -1;
    
    for (size_t i = 0; i < newShape.size(); i++) {
        if (newShape[i] == -1) {
            if (inferDim != -1) {
                throw std::runtime_error("Only one dimension can be inferred");
            }
            inferDim = i;
        } else {
            newSize *= newShape[i];
        }
    }
    
    std::vector<int> actualShape = newShape;
    if (inferDim != -1) {
        actualShape[inferDim] = m_size / newSize;
        newSize *= actualShape[inferDim];
    }
    
    if (newSize != m_size) {
        throw std::runtime_error("New shape is incompatible with tensor size");
    }
    
    return Tensor(m_data, actualShape);
}

inline Tensor Tensor::flatten() const {
    return reshape({m_size});
}

inline Tensor Tensor::operator+(const Tensor& other) const {
    if (!isCompatible(other)) {
        throw std::runtime_error("Tensor shapes are incompatible");
    }
    
    Tensor result(m_shape);
    for (int i = 0; i < m_size; i++) {
        result.m_data[i] = m_data[i] + other.m_data[i];
    }
    return result;
}

inline Tensor Tensor::operator-(const Tensor& other) const {
    if (!isCompatible(other)) {
        throw std::runtime_error("Tensor shapes are incompatible");
    }
    
    Tensor result(m_shape);
    for (int i = 0; i < m_size; i++) {
        result.m_data[i] = m_data[i] - other.m_data[i];
    }
    return result;
}

inline Tensor Tensor::operator*(const Tensor& other) const {
    if (!isCompatible(other)) {
        throw std::runtime_error("Tensor shapes are incompatible");
    }
    
    Tensor result(m_shape);
    for (int i = 0; i < m_size; i++) {
        result.m_data[i] = m_data[i] * other.m_data[i];
    }
    return result;
}

inline Tensor Tensor::operator*(float scalar) const {
    Tensor result(m_shape);
    for (int i = 0; i < m_size; i++) {
        result.m_data[i] = m_data[i] * scalar;
    }
    return result;
}

inline Tensor Tensor::operator+(float scalar) const {
    Tensor result(m_shape);
    for (int i = 0; i < m_size; i++) {
        result.m_data[i] = m_data[i] + scalar;
    }
    return result;
}

inline Tensor Tensor::matmul(const Tensor& other) const {
    if (m_shape.size() != 2 || other.m_shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication requires 2D tensors");
    }
    
    int m = m_shape[0];
    int k = m_shape[1];
    int n = other.m_shape[1];
    
    if (k != other.m_shape[0]) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    Tensor result({m, n}, 0.0f);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += m_data[i * k + p] * other.m_data[p * n + j];
            }
            result.m_data[i * n + j] = sum;
        }
    }
    
    return result;
}

inline Tensor Tensor::transpose() const {
    if (m_shape.size() != 2) {
        throw std::runtime_error("Transpose only supports 2D tensors");
    }
    
    int rows = m_shape[0];
    int cols = m_shape[1];
    
    Tensor result({cols, rows});
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.m_data[j * rows + i] = m_data[i * cols + j];
        }
    }
    
    return result;
}

inline Tensor Tensor::relu() const {
    Tensor result(m_shape);
    for (int i = 0; i < m_size; i++) {
        result.m_data[i] = std::max(0.0f, m_data[i]);
    }
    return result;
}

inline Tensor Tensor::leakyRelu(float alpha) const {
    Tensor result(m_shape);
    for (int i = 0; i < m_size; i++) {
        result.m_data[i] = m_data[i] > 0 ? m_data[i] : alpha * m_data[i];
    }
    return result;
}

inline Tensor Tensor::sigmoid() const {
    Tensor result(m_shape);
    for (int i = 0; i < m_size; i++) {
        result.m_data[i] = 1.0f / (1.0f + std::exp(-m_data[i]));
    }
    return result;
}

inline Tensor Tensor::tanh() const {
    Tensor result(m_shape);
    for (int i = 0; i < m_size; i++) {
        result.m_data[i] = std::tanh(m_data[i]);
    }
    return result;
}

inline float Tensor::sum() const {
    return std::accumulate(m_data.begin(), m_data.end(), 0.0f);
}

inline float Tensor::mean() const {
    return sum() / m_size;
}

inline float Tensor::max() const {
    return *std::max_element(m_data.begin(), m_data.end());
}

inline float Tensor::min() const {
    return *std::min_element(m_data.begin(), m_data.end());
}

inline void Tensor::fill(float value) {
    std::fill(m_data.begin(), m_data.end(), value);
}

inline void Tensor::zero() {
    fill(0.0f);
}

inline void Tensor::ones() {
    fill(1.0f);
}

inline void Tensor::randomUniform(float min, float max, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min, max);
    
    for (float& val : m_data) {
        val = dist(gen);
    }
}

inline void Tensor::randomNormal(float mean, float std, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(mean, std);
    
    for (float& val : m_data) {
        val = dist(gen);
    }
}

inline void Tensor::xavier(int fanIn, int fanOut, int seed) {
    float limit = std::sqrt(6.0f / (fanIn + fanOut));
    randomUniform(-limit, limit, seed);
}

inline void Tensor::he(int fanIn, int seed) {
    float std = std::sqrt(2.0f / fanIn);
    randomNormal(0.0f, std, seed);
}

inline bool Tensor::isCompatible(const Tensor& other) const {
    return m_shape == other.m_shape;
}

inline void Tensor::print() const {
    // Simple print for debugging
    for (int i = 0; i < std::min(10, m_size); i++) {
        // Print first 10 elements
    }
}

} // namespace nn
} // namespace mcgan