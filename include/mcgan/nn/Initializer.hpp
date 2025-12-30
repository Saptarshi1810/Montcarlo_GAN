#pragma once

#include "Tensor.hpp"
#include <random>
#include <cmath>

namespace mcgan {
namespace nn {

/**
 * Weight initialization strategies for neural networks.
 */
class Initializer {
public:
    virtual ~Initializer() = default;
    
    /**
     * Initialize a tensor with the given strategy.
     */
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) = 0;
};

/**
 * Xavier/Glorot uniform initialization.
 * Uniform distribution in [-limit, limit]
 * where limit = sqrt(6 / (fan_in + fan_out))
 */
class XavierUniform : public Initializer {
public:
    XavierUniform() = default;
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        float limit = std::sqrt(6.0f / (fanIn + fanOut));
        tensor.randomUniform(-limit, limit, seed);
    }
};

/**
 * Xavier/Glorot normal initialization.
 * Normal distribution with std = sqrt(2 / (fan_in + fan_out))
 */
class XavierNormal : public Initializer {
public:
    XavierNormal() = default;
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        float std = std::sqrt(2.0f / (fanIn + fanOut));
        tensor.randomNormal(0.0f, std, seed);
    }
};

/**
 * He/Kaiming uniform initialization.
 * Uniform distribution in [-limit, limit]
 * where limit = sqrt(6 / fan_in)
 */
class HeUniform : public Initializer {
public:
    HeUniform() = default;
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        float limit = std::sqrt(6.0f / fanIn);
        tensor.randomUniform(-limit, limit, seed);
    }
};

/**
 * He/Kaiming normal initialization.
 * Normal distribution with std = sqrt(2 / fan_in)
 * Recommended for ReLU activations.
 */
class HeNormal : public Initializer {
public:
    HeNormal() = default;
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        float std = std::sqrt(2.0f / fanIn);
        tensor.randomNormal(0.0f, std, seed);
    }
};

/**
 * LeCun uniform initialization.
 * Uniform distribution in [-limit, limit]
 * where limit = sqrt(3 / fan_in)
 */
class LeCunUniform : public Initializer {
public:
    LeCunUniform() = default;
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        float limit = std::sqrt(3.0f / fanIn);
        tensor.randomUniform(-limit, limit, seed);
    }
};

/**
 * LeCun normal initialization.
 * Normal distribution with std = sqrt(1 / fan_in)
 * Recommended for SELU activations.
 */
class LeCunNormal : public Initializer {
public:
    LeCunNormal() = default;
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        float std = std::sqrt(1.0f / fanIn);
        tensor.randomNormal(0.0f, std, seed);
    }
};

/**
 * Uniform initialization in a given range.
 */
class Uniform : public Initializer {
public:
    Uniform(float min = -0.1f, float max = 0.1f) 
        : m_min(min), m_max(max) {}
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        tensor.randomUniform(m_min, m_max, seed);
    }

private:
    float m_min;
    float m_max;
};

/**
 * Normal/Gaussian initialization.
 */
class Normal : public Initializer {
public:
    Normal(float mean = 0.0f, float std = 0.01f) 
        : m_mean(mean), m_std(std) {}
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        tensor.randomNormal(m_mean, m_std, seed);
    }

private:
    float m_mean;
    float m_std;
};

/**
 * Constant initialization.
 */
class Constant : public Initializer {
public:
    Constant(float value = 0.0f) : m_value(value) {}
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        tensor.fill(m_value);
    }

private:
    float m_value;
};

/**
 * Zero initialization.
 */
class Zeros : public Initializer {
public:
    Zeros() = default;
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        tensor.zero();
    }
};

/**
 * One initialization.
 */
class Ones : public Initializer {
public:
    Ones() = default;
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        tensor.ones();
    }
};

/**
 * Orthogonal initialization.
 * Initializes with an orthogonal matrix (useful for RNNs).
 */
class Orthogonal : public Initializer {
public:
    Orthogonal(float gain = 1.0f) : m_gain(gain) {}
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
        if (tensor.ndim() != 2) {
            throw std::runtime_error("Orthogonal initialization requires 2D tensor");
        }
        
        int rows = tensor.shape()[0];
        int cols = tensor.shape()[1];
        
        // Generate random matrix
        tensor.randomNormal(0.0f, 1.0f, seed);
        
        // QR decomposition (simplified - full implementation would use proper QR)
        // For now, use normalized random initialization with gain
        float scale = m_gain / std::sqrt(static_cast<float>(std::min(rows, cols)));
        
        for (int i = 0; i < tensor.size(); i++) {
            tensor[i] *= scale;
        }
    }

private:
    float m_gain;
};

/**
 * Sparse initialization.
 * Initializes with a given sparsity (percentage of zeros).
 */
class Sparse : public Initializer {
public:
    Sparse(float sparsity = 0.1f, float std = 0.01f) 
        : m_sparsity(sparsity), m_std(std) {}
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
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

private:
    float m_sparsity;
    float m_std;
};

/**
 * Identity initialization.
 * Initializes as identity matrix (diagonal ones).
 */
class Identity : public Initializer {
public:
    Identity(float gain = 1.0f) : m_gain(gain) {}
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
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

private:
    float m_gain;
};

/**
 * Variance Scaling initialization.
 * Generalizes Xavier and He initialization.
 */
class VarianceScaling : public Initializer {
public:
    enum class Mode {
        FAN_IN,
        FAN_OUT,
        FAN_AVG
    };
    
    enum class Distribution {
        UNIFORM,
        NORMAL
    };
    
    VarianceScaling(float scale = 1.0f, Mode mode = Mode::FAN_IN, 
                   Distribution distribution = Distribution::NORMAL)
        : m_scale(scale), m_mode(mode), m_distribution(distribution) {}
    
    virtual void initialize(Tensor& tensor, int fanIn, int fanOut, int seed = 0) override {
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

private:
    float m_scale;
    Mode m_mode;
    Distribution m_distribution;
};

/**
 * Factory function to create initializers.
 */
inline std::unique_ptr<Initializer> createInitializer(const std::string& type) {
    if (type == "xavier_uniform") {
        return std::make_unique<XavierUniform>();
    } else if (type == "xavier_normal") {
        return std::make_unique<XavierNormal>();
    } else if (type == "he_uniform") {
        return std::make_unique<HeUniform>();
    } else if (type == "he_normal") {
        return std::make_unique<HeNormal>();
    } else if (type == "lecun_uniform") {
        return std::make_unique<LeCunUniform>();
    } else if (type == "lecun_normal") {
        return std::make_unique<LeCunNormal>();
    } else if (type == "zeros") {
        return std::make_unique<Zeros>();
    } else if (type == "ones") {
        return std::make_unique<Ones>();
    } else if (type == "orthogonal") {
        return std::make_unique<Orthogonal>();
    } else {
        return std::make_unique<XavierNormal>();  // Default
    }
}

} // namespace nn
} // namespace mcgan