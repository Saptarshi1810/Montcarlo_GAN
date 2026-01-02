#pragma once

#include "Tensor.hpp"
#include <vector>
#include <memory>
#include <cmath>

namespace mcgan {
namespace nn {

/**
 * Base class for optimizers.
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    /**
     * Perform one optimization step.
     */
    virtual void step() = 0;
    
    /**
     * Zero out all gradients.
     */
    virtual void zeroGrad() = 0;
    
    /**
     * Get learning rate.
     */
    virtual float getLearningRate() const = 0;
    
    /**
     * Set learning rate.
     */
    virtual void setLearningRate(float lr) = 0;
};

/**
 * Stochastic Gradient Descent optimizer.
 */
class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> parameters, std::vector<Tensor*> gradients,
        float lr = 0.01f, float momentum = 0.0f, float weightDecay = 0.0f)
        : m_parameters(parameters)
        , m_gradients(gradients)
        , m_lr(lr)
        , m_momentum(momentum)
        , m_weightDecay(weightDecay)
    {
        if (m_momentum > 0.0f) {
            m_velocities.resize(m_parameters.size());
            for (size_t i = 0; i < m_parameters.size(); i++) {
                m_velocities[i] = Tensor(m_parameters[i]->shape(), 0.0f);
            }
        }
    }
    
    virtual void step() override {
        for (size_t i = 0; i < m_parameters.size(); i++) {
            Tensor* param = m_parameters[i];
            Tensor* grad = m_gradients[i];
            
            // Weight decay
            if (m_weightDecay > 0.0f) {
                for (int j = 0; j < param->size(); j++) {
                    (*grad)[j] += m_weightDecay * (*param)[j];
                }
            }
            
            // Momentum
            if (m_momentum > 0.0f) {
                for (int j = 0; j < param->size(); j++) {
                    m_velocities[i][j] = m_momentum * m_velocities[i][j] - m_lr * (*grad)[j];
                    (*param)[j] += m_velocities[i][j];
                }
            } else {
                // Standard SGD
                for (int j = 0; j < param->size(); j++) {
                    (*param)[j] -= m_lr * (*grad)[j];
                }
            }
        }
    }
    
    virtual void zeroGrad() override {
        for (Tensor* grad : m_gradients) {
            grad->zero();
        }
    }
    
    virtual float getLearningRate() const override { return m_lr; }
    virtual void setLearningRate(float lr) override { m_lr = lr; }

private:
    std::vector<Tensor*> m_parameters;
    std::vector<Tensor*> m_gradients;
    std::vector<Tensor> m_velocities;
    float m_lr;
    float m_momentum;
    float m_weightDecay;
};

/**
 * Adam optimizer (Adaptive Moment Estimation).
 */
class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> parameters, std::vector<Tensor*> gradients,
         float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, 
         float epsilon = 1e-8f, float weightDecay = 0.0f)
        : m_parameters(parameters)
        , m_gradients(gradients)
        , m_lr(lr)
        , m_beta1(beta1)
        , m_beta2(beta2)
        , m_epsilon(epsilon)
        , m_weightDecay(weightDecay)
        , m_t(0)
    {
        // Initialize moment vectors
        m_m.resize(m_parameters.size());
        m_v.resize(m_parameters.size());
        
        for (size_t i = 0; i < m_parameters.size(); i++) {
            m_m[i] = Tensor(m_parameters[i]->shape(), 0.0f);
            m_v[i] = Tensor(m_parameters[i]->shape(), 0.0f);
        }
    }
    
    virtual void step() override {
        m_t++;
        
        float bias1 = 1.0f - std::pow(m_beta1, m_t);
        float bias2 = 1.0f - std::pow(m_beta2, m_t);
        
        for (size_t i = 0; i < m_parameters.size(); i++) {
            Tensor* param = m_parameters[i];
            Tensor* grad = m_gradients[i];
            
            for (int j = 0; j < param->size(); j++) {
                float g = (*grad)[j];
                
                // Weight decay
                if (m_weightDecay > 0.0f) {
                    g += m_weightDecay * (*param)[j];
                }
                
                // Update biased first moment estimate
                m_m[i][j] = m_beta1 * m_m[i][j] + (1.0f - m_beta1) * g;
                
                // Update biased second raw moment estimate
                m_v[i][j] = m_beta2 * m_v[i][j] + (1.0f - m_beta2) * g * g;
                
                // Compute bias-corrected moments
                float m_hat = m_m[i][j] / bias1;
                float v_hat = m_v[i][j] / bias2;
                
                // Update parameters
                (*param)[j] -= m_lr * m_hat / (std::sqrt(v_hat) + m_epsilon);
            }
        }
    }
    
    virtual void zeroGrad() override {
        for (Tensor* grad : m_gradients) {
            grad->zero();
        }
    }
    
    virtual float getLearningRate() const override { return m_lr; }
    virtual void setLearningRate(float lr) override { m_lr = lr; }
    
    void reset() {
        m_t = 0;
        for (size_t i = 0; i < m_m.size(); i++) {
            m_m[i].zero();
            m_v[i].zero();
        }
    }

private:
    std::vector<Tensor*> m_parameters;
    std::vector<Tensor*> m_gradients;
    std::vector<Tensor> m_m;  // First moment
    std::vector<Tensor> m_v;  // Second moment
    float m_lr;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    float m_weightDecay;
    int m_t;  // Timestep
};

/**
 * RMSprop optimizer.
 */
class RMSprop : public Optimizer {
public:
    RMSprop(std::vector<Tensor*> parameters, std::vector<Tensor*> gradients,
            float lr = 0.01f, float alpha = 0.99f, float epsilon = 1e-8f,
            float weightDecay = 0.0f)
        : m_parameters(parameters)
        , m_gradients(gradients)
        , m_lr(lr)
        , m_alpha(alpha)
        , m_epsilon(epsilon)
        , m_weightDecay(weightDecay)
    {
        m_v.resize(m_parameters.size());
        
        for (size_t i = 0; i < m_parameters.size(); i++) {
            m_v[i] = Tensor(m_parameters[i]->shape(), 0.0f);
        }
    }
    
    virtual void step() override {
        for (size_t i = 0; i < m_parameters.size(); i++) {
            Tensor* param = m_parameters[i];
            Tensor* grad = m_gradients[i];
            
            for (int j = 0; j < param->size(); j++) {
                float g = (*grad)[j];
                
                // Weight decay
                if (m_weightDecay > 0.0f) {
                    g += m_weightDecay * (*param)[j];
                }
                
                // Update moving average of squared gradient
                m_v[i][j] = m_alpha * m_v[i][j] + (1.0f - m_alpha) * g * g;
                
                // Update parameters
                (*param)[j] -= m_lr * g / (std::sqrt(m_v[i][j]) + m_epsilon);
            }
        }
    }
    
    virtual void zeroGrad() override {
        for (Tensor* grad : m_gradients) {
            grad->zero();
        }
    }
    
    virtual float getLearningRate() const override { return m_lr; }
    virtual void setLearningRate(float lr) override { m_lr = lr; }

private:
    std::vector<Tensor*> m_parameters;
    std::vector<Tensor*> m_gradients;
    std::vector<Tensor> m_v;  // Moving average of squared gradients
    float m_lr;
    float m_alpha;
    float m_epsilon;
    float m_weightDecay;
};

/**
 * AdaGrad optimizer.
 */
class AdaGrad : public Optimizer {
public:
    AdaGrad(std::vector<Tensor*> parameters, std::vector<Tensor*> gradients,
            float lr = 0.01f, float epsilon = 1e-8f, float weightDecay = 0.0f)
        : m_parameters(parameters)
        , m_gradients(gradients)
        , m_lr(lr)
        , m_epsilon(epsilon)
        , m_weightDecay(weightDecay)
    {
        m_sumSquaredGrad.resize(m_parameters.size());
        
        for (size_t i = 0; i < m_parameters.size(); i++) {
            m_sumSquaredGrad[i] = Tensor(m_parameters[i]->shape(), 0.0f);
        }
    }
    
    virtual void step() override {
        for (size_t i = 0; i < m_parameters.size(); i++) {
            Tensor* param = m_parameters[i];
            Tensor* grad = m_gradients[i];
            
            for (int j = 0; j < param->size(); j++) {
                float g = (*grad)[j];
                
                // Weight decay
                if (m_weightDecay > 0.0f) {
                    g += m_weightDecay * (*param)[j];
                }
                
                // Accumulate squared gradients
                m_sumSquaredGrad[i][j] += g * g;
                
                // Update parameters
                (*param)[j] -= m_lr * g / (std::sqrt(m_sumSquaredGrad[i][j]) + m_epsilon);
            }
        }
    }
    
    virtual void zeroGrad() override {
        for (Tensor* grad : m_gradients) {
            grad->zero();
        }
    }
    
    virtual float getLearningRate() const override { return m_lr; }
    virtual void setLearningRate(float lr) override { m_lr = lr; }

private:
    std::vector<Tensor*> m_parameters;
    std::vector<Tensor*> m_gradients;
    std::vector<Tensor> m_sumSquaredGrad;
    float m_lr;
    float m_epsilon;
    float m_weightDecay;
};

/**
 * Learning rate scheduler.
 */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    
    /**
     * Update learning rate based on epoch/step.
     */
    virtual void step(int epoch) = 0;
    
    /**
     * Get current learning rate.
     */
    virtual float getLearningRate() const = 0;
};

/**
 * Step decay learning rate scheduler.
 */
class StepLR : public LRScheduler {
public:
    StepLR(Optimizer* optimizer, int stepSize, float gamma = 0.1f)
        : m_optimizer(optimizer)
        , m_stepSize(stepSize)
        , m_gamma(gamma)
        , m_baseLR(optimizer->getLearningRate())
    {}
    
    virtual void step(int epoch) override {
        int numDecays = epoch / m_stepSize;
        float newLR = m_baseLR * std::pow(m_gamma, numDecays);
        m_optimizer->setLearningRate(newLR);
    }
    
    virtual float getLearningRate() const override {
        return m_optimizer->getLearningRate();
    }

private:
    Optimizer* m_optimizer;
    int m_stepSize;
    float m_gamma;
    float m_baseLR;
};

/**
 * Exponential decay learning rate scheduler.
 */
class ExponentialLR : public LRScheduler {
public:
    ExponentialLR(Optimizer* optimizer, float gamma = 0.95f)
        : m_optimizer(optimizer)
        , m_gamma(gamma)
        , m_baseLR(optimizer->getLearningRate())
    {}
    
    virtual void step(int epoch) override {
        float newLR = m_baseLR * std::pow(m_gamma, epoch);
        m_optimizer->setLearningRate(newLR);
    }
    
    virtual float getLearningRate() const override {
        return m_optimizer->getLearningRate();
    }

private:
    Optimizer* m_optimizer;
    float m_gamma;
    float m_baseLR;
};

/**
 * Cosine annealing learning rate scheduler.
 */
class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer* optimizer, int maxEpochs, float minLR = 0.0f)
        : m_optimizer(optimizer)
        , m_maxEpochs(maxEpochs)
        , m_minLR(minLR)
        , m_baseLR(optimizer->getLearningRate())
    {}
    
    virtual void step(int epoch) override {
        float cosine = std::cos(M_PI * epoch / m_maxEpochs);
        float newLR = m_minLR + (m_baseLR - m_minLR) * (1.0f + cosine) / 2.0f;
        m_optimizer->setLearningRate(newLR);
    }
    
    virtual float getLearningRate() const override {
        return m_optimizer->getLearningRate();
    }

private:
    Optimizer* m_optimizer;
    int m_maxEpochs;
    float m_minLR;
    float m_baseLR;
};

} // namespace nn
} // namespace mcgan