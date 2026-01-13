#include "Initializer.hpp"
#include <cmath>
#include <random>
#include <algorithm>

namespace mcgan {
namespace nn {

// ============================================================================
// Xavier/Glorot Initializers
// ============================================================================

void XavierUniform::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    float limit = std::sqrt(6.0f / (fanIn + fanOut));
    tensor.randomUniform(-limit, limit, seed);
}

void XavierNormal::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    float std = std::sqrt(2.0f / (fanIn + fanOut));
    tensor.randomNormal(0.0f, std, seed);
}

// ============================================================================
// He/Kaiming Initializers
// ============================================================================

void HeUniform::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    float limit = std::sqrt(6.0f / fanIn);
    tensor.randomUniform(-limit, limit, seed);
}

void HeNormal::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    float std = std::sqrt(2.0f / fanIn);
    tensor.randomNormal(0.0f, std, seed);
}

// ============================================================================
// LeCun Initializers
// ============================================================================

void LeCunUniform::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    float limit = std::sqrt(3.0f / fanIn);
    tensor.randomUniform(-limit, limit, seed);
}

void LeCunNormal::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    float std = std::sqrt(1.0f / fanIn);
    tensor.randomNormal(0.0f, std, seed);
}

// ============================================================================
// Simple Initializers
// ============================================================================

void Uniform::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    tensor.randomUniform(m_min, m_max, seed);
}

void Normal::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    tensor.randomNormal(m_mean, m_std, seed);
}

void Constant::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    tensor.fill(m_value);
}

void Zeros::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    tensor.zero();
}

void Ones::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    tensor.ones();
}

// ============================================================================
// Orthogonal Initializer
// ============================================================================

void Orthogonal::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    if (tensor.ndim() != 2) {
        throw std::runtime_error("Orthogonal initialization requires 2D tensor");
    }
    
    int rows = tensor.shape()[0];
    int cols = tensor.shape()[1];
    
    // Generate random Gaussian matrix
    tensor.randomNormal(0.0f, 1.0f, seed);
    
    // Perform simplified QR decomposition
    // For a proper implementation, use a full QR decomposition algorithm
    // This is a simplified version using Gram-Schmidt orthogonalization
    
    if (rows >= cols) {
        // Orthogonalize columns
        for (int j = 0; j < cols; j++) {
            // Normalize current column
            float norm = 0.0f;
            for (int i = 0; i < rows; i++) {
                float val = tensor[i * cols + j];
                norm += val * val;
            }
            norm = std::sqrt(norm);
            
            if (norm > 1e-8f) {
                for (int i = 0; i < rows; i++) {
                    tensor[i * cols + j] /= norm;
                }
            }
            
            // Orthogonalize remaining columns against this one
            for (int k = j + 1; k < cols; k++) {
                float dot = 0.0f;
                for (int i = 0; i < rows; i++) {
                    dot += tensor[i * cols + j] * tensor[i * cols + k];
                }
                
                for (int i = 0; i < rows; i++) {
                    tensor[i * cols + k] -= dot * tensor[i * cols + j];
                }
            }
        }
    } else {
        // Orthogonalize rows
        for (int i = 0; i < rows; i++) {
            // Normalize current row
            float norm = 0.0f;
            for (int j = 0; j < cols; j++) {
                float val = tensor[i * cols + j];
                norm += val * val;
            }
            norm = std::sqrt(norm);
            
            if (norm > 1e-8f) {
                for (int j = 0; j < cols; j++) {
                    tensor[i * cols + j] /= norm;
                }
            }
            
            // Orthogonalize remaining rows against this one
            for (int k = i + 1; k < rows; k++) {
                float dot = 0.0f;
                for (int j = 0; j < cols; j++) {
                    dot += tensor[i * cols + j] * tensor[k * cols + j];
                }
                
                for (int j = 0; j < cols; j++) {
                    tensor[k * cols + j] -= dot * tensor[i * cols + j];
                }
            }
        }
    }
    
    // Apply gain
    if (m_gain != 1.0f) {
        for (int i = 0; i < tensor.size(); i++) {
            tensor[i] *= m_gain;
        }
    }
}

// ============================================================================
// Sparse Initializer
// ============================================================================

void Sparse::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> normalDist(0.0f, m_std);
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
    
    for (int i = 0; i < tensor.size(); i++) {
        if (uniformDist(gen) < m_sparsity) {
            tensor[i] = 0.0f;
        } else {
            tensor[i] = normalDist(gen);
        }
    }
}

// ============================================================================
// Identity Initializer
// ============================================================================

void Identity::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    if (tensor.ndim() != 2) {
        throw std::runtime_error("Identity initialization requires 2D tensor");
    }
    
    tensor.zero();
    
    int rows = tensor.shape()[0];
    int cols = tensor.shape()[1];
    int minDim = std::min(rows, cols);
    
    for (int i = 0; i < minDim; i++) {
        tensor[i * cols + i] = m_gain;
    }
}

// ============================================================================
// Variance Scaling Initializer
// ============================================================================

void VarianceScaling::initialize(Tensor& tensor, int fanIn, int fanOut, int seed) {
    float fan = 0.0f;
    
    switch (m_mode) {
        case Mode::FAN_IN:
            fan = static_cast<float>(fanIn);
            break;
        case Mode::FAN_OUT:
            fan = static_cast<float>(fanOut);
            break;
        case Mode::FAN_AVG:
            fan = (fanIn + fanOut) / 2.0f;
            break;
    }
    
    if (m_distribution == Distribution::UNIFORM) {
        float limit = std::sqrt(3.0f * m_scale / fan);
        tensor.randomUniform(-limit, limit, seed);
    } else {
        float std = std::sqrt(m_scale / fan);
        tensor.randomNormal(0.0f, std, seed);
    }
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<Initializer> createInitializer(const std::string& type) {
    if (type == "xavier_uniform" || type == "glorot_uniform") {
        return std::make_unique<XavierUniform>();
    } else if (type == "xavier_normal" || type == "glorot_normal") {
        return std::make_unique<XavierNormal>();
    } else if (type == "he_uniform" || type == "kaiming_uniform") {
        return std::make_unique<HeUniform>();
    } else if (type == "he_normal" || type == "kaiming_normal") {
        return std::make_unique<HeNormal>();
    } else if (type == "lecun_uniform") {
        return std::make_unique<LeCunUniform>();
    } else if (type == "lecun_normal") {
        return std::make_unique<LeCunNormal>();
    } else if (type == "zeros" || type == "zero") {
        return std::make_unique<Zeros>();
    } else if (type == "ones" || type == "one") {
        return std::make_unique<Ones>();
    } else if (type == "orthogonal") {
        return std::make_unique<Orthogonal>();
    } else if (type == "identity") {
        return std::make_unique<Identity>();
    } else if (type == "uniform") {
        return std::make_unique<Uniform>();
    } else if (type == "normal" || type == "gaussian") {
        return std::make_unique<Normal>();
    } else {
        // Default to Xavier Normal
        return std::make_unique<XavierNormal>();
    }
}

} // namespace nn
} // namespace mcgan