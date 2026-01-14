#include "Optimizer.hpp"
#include <cmath>
#include <algorithm>

namespace mcgan {
namespace nn {

// ============================================================================
// SGD Optimizer
// ============================================================================

SGD::SGD(std::vector<Tensor*> parameters, std::vector<Tensor*> gradients,
         float lr, float momentum, float weightDecay)
    : m_parameters(parameters)
    , m_gradients(gradients)
    , m_lr(lr)
    , m_momentum(momentum)
    , m_weightDecay(weightDecay)
{
    if (m_parameters.size() != m_gradients.size()) {
        throw std::runtime_error("Parameters and gradients must have the same size");
    }
    
    if (m_momentum > 0.0f) {
        m_velocities.resize(m_parameters.size());
        for (size_t i = 0; i < m_parameters.size(); i++) {
            m_velocities[i] = Tensor(m_parameters[i]->shape(), 0.0f);
        }
    }
}

void SGD::step() {
    for (size_t i = 0; i < m_parameters.size(); i++) {
        Tensor* param = m_parameters[i];
        Tensor* grad = m_gradients[i];
        
        // Apply weight decay (L2 regularization)
        if (m_weightDecay > 0.0f) {
            for (int j = 0; j < param->size(); j++) {
                (*grad)[j] += m_weightDecay * (*param)[j];
            }
        }
        
        // Update with momentum
        if (m_momentum > 0.0f) {
            for (int j = 0; j < param->size(); j++) {
                m_velocities[i][j] = m_momentum * m_velocities[i][j] - m_lr * (*grad)[j];
                (*param)[j] += m_velocities[i][j];
            }
        } else {
            // Standard SGD: θ = θ - lr * ∇θ
            for (int j = 0; j < param->size(); j++) {
                (*param)[j] -= m_lr * (*grad)[j];
            }
        }
    }
}

void SGD::zeroGrad() {
    for (Tensor* grad : m_gradients) {
        grad->zero();
    }
}

// ============================================================================
// Adam Optimizer
// ============================================================================

Adam::Adam(std::vector<Tensor*> parameters, std::vector<Tensor*> gradients,
           float lr, float beta1, float beta2, float epsilon, float weightDecay)
    : m_parameters(parameters)
    , m_gradients(gradients)
    , m_lr(lr)
    , m_beta1(beta1)
    , m_beta2(beta2)
    , m_epsilon(epsilon)
    , m_weightDecay(weightDecay)
    , m_t(0)
{
    if (m_parameters.size() != m_gradients.size()) {
        throw std::runtime_error("Parameters and gradients must have the same size");
    }
    
    // Initialize first and second moment vectors
    m_m.resize(m_parameters.size());
    m_v.resize(m_parameters.size());
    
    for (size_t i = 0; i < m_parameters.size(); i++) {
        m_m[i] = Tensor(m_parameters[i]->shape(), 0.0f);
        m_v[i] = Tensor(m_parameters[i]->shape(), 0.0f);
    }
}

void Adam::step() {
    m_t++;
    
    // Compute bias correction terms
    float bias1 = 1.0f - std::pow(m_beta1, static_cast<float>(m_t));
    float bias2 = 1.0f - std::pow(m_beta2, static_cast<float>(m_t));
    
    for (size_t i = 0; i < m_parameters.size(); i++) {
        Tensor* param = m_parameters[i];
        Tensor* grad = m_gradients[i];
        
        for (int j = 0; j < param->size(); j++) {
            float g = (*grad)[j];
            
            // Apply weight decay (AdamW variant)
            if (m_weightDecay > 0.0f) {
                g += m_weightDecay * (*param)[j];
            }
            
            // Update biased first moment estimate: m_t = β1 * m_{t-1} + (1-β1) * g_t
            m_m[i][j] = m_beta1 * m_m[i][j] + (1.0f - m_beta1) * g;
            
            // Update biased second raw moment estimate: v_t = β2 * v_{t-1} + (1-β2) * g_t^2
            m_v[i][j] = m_beta2 * m_v[i][j] + (1.0f - m_beta2) * g * g;
            
            // Compute bias-corrected first moment estimate
            float m_hat = m_m[i][j] / bias1;
            
            // Compute bias-corrected second raw moment estimate
            float v_hat = m_v[i][j] / bias2;
            
            // Update parameters: θ_t = θ_{t-1} - α * m_hat / (√v_hat + ε)
            (*param)[j] -= m_lr * m_hat / (std::sqrt(v_hat) + m_epsilon);
        }
    }
}

void Adam::zeroGrad() {
    for (Tensor* grad : m_gradients) {
        grad->zero();
    }
}

void Adam::reset() {
    m_t = 0;
    for (size_t i = 0; i < m_m.size(); i++) {
        m_m[i].zero();
        m_v[i].zero();
    }
}

// ============================================================================
// RMSprop Optimizer
// ============================================================================

RMSprop::RMSprop(std::vector<Tensor*> parameters, std::vector<Tensor*> gradients,
                 float lr, float alpha, float epsilon, float weightDecay)
    : m_parameters(parameters)
    , m_gradients(gradients)
    , m_lr(lr)
    , m_alpha(alpha)
    , m_epsilon(epsilon)
    , m_weightDecay(weightDecay)
{
    if (m_parameters.size() != m_gradients.size()) {
        throw std::runtime_error("Parameters and gradients must have the same size");
    }
    
    m_v.resize(m_parameters.size());
    
    for (size_t i = 0; i < m_parameters.size(); i++) {
        m_v[i] = Tensor(m_parameters[i]->shape(), 0.0f);
    }
}

void RMSprop::step() {
    for (size_t i = 0; i < m_parameters.size(); i++) {
        Tensor* param = m_parameters[i];
        Tensor* grad = m_gradients[i];
        
        for (int j = 0; j < param->size(); j++) {
            float g = (*grad)[j];
            
            // Apply weight decay
            if (m_weightDecay > 0.0f) {
                g += m_weightDecay * (*param)[j];
            }
            
            // Update moving average of squared gradient: v_t = α * v_{t-1} + (1-α) * g_t^2
            m_v[i][j] = m_alpha * m_v[i][j] + (1.0f - m_alpha) * g * g;
            
            // Update parameters: θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)
            (*param)[j] -= m_lr * g / (std::sqrt(m_v[i][j]) + m_epsilon);
        }
    }
}

void RMSprop::zeroGrad() {
    for (Tensor* grad : m_gradients) {
        grad->zero();
    }
}

// ============================================================================
// AdaGrad Optimizer
// ============================================================================

AdaGrad::AdaGrad(std::vector<Tensor*> parameters, std::vector<Tensor*> gradients,
                 float lr, float epsilon, float weightDecay)
    : m_parameters(parameters)
    , m_gradients(gradients)
    , m_lr(lr)
    , m_epsilon(epsilon)
    , m_weightDecay(weightDecay)
{
    if (m_parameters.size() != m_gradients.size()) {
        throw std::runtime_error("Parameters and gradients must have the same size");
    }
    
    m_sumSquaredGrad.resize(m_parameters.size());
    
    for (size_t i = 0; i < m_parameters.size(); i++) {
        m_sumSquaredGrad[i] = Tensor(m_parameters[i]->shape(), 0.0f);
    }
}

void AdaGrad::step() {
    for (size_t i = 0; i < m_parameters.size(); i++) {
        Tensor* param = m_parameters[i];
        Tensor* grad = m_gradients[i];
        
        for (int j = 0; j < param->size(); j++) {
            float g = (*grad)[j];
            
            // Apply weight decay
            if (m_weightDecay > 0.0f) {
                g += m_weightDecay * (*param)[j];
            }
            
            // Accumulate squared gradients: G_t = G_{t-1} + g_t^2
            m_sumSquaredGrad[i][j] += g * g;
            
            // Update parameters: θ_t = θ_{t-1} - lr * g_t / (√G_t + ε)
            (*param)[j] -= m_lr * g / (std::sqrt(m_sumSquaredGrad[i][j]) + m_epsilon);
        }
    }
}

void AdaGrad::zeroGrad() {
    for (Tensor* grad : m_gradients) {
        grad->zero();
    }
}

// ============================================================================
// Learning Rate Schedulers
// ============================================================================

StepLR::StepLR(Optimizer* optimizer, int stepSize, float gamma)
    : m_optimizer(optimizer)
    , m_stepSize(stepSize)
    , m_gamma(gamma)
    , m_baseLR(optimizer->getLearningRate())
{
}

void StepLR::step(int epoch) {
    int numDecays = epoch / m_stepSize;
    float newLR = m_baseLR * std::pow(m_gamma, static_cast<float>(numDecays));
    m_optimizer->setLearningRate(newLR);
}

ExponentialLR::ExponentialLR(Optimizer* optimizer, float gamma)
    : m_optimizer(optimizer)
    , m_gamma(gamma)
    , m_baseLR(optimizer->getLearningRate())
{
}

void ExponentialLR::step(int epoch) {
    float newLR = m_baseLR * std::pow(m_gamma, static_cast<float>(epoch));
    m_optimizer->setLearningRate(newLR);
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer* optimizer, int maxEpochs, float minLR)
    : m_optimizer(optimizer)
    , m_maxEpochs(maxEpochs)
    , m_minLR(minLR)
    , m_baseLR(optimizer->getLearningRate())
{
}

void CosineAnnealingLR::step(int epoch) {
    float cosine = std::cos(M_PI * epoch / m_maxEpochs);
    float newLR = m_minLR + (m_baseLR - m_minLR) * (1.0f + cosine) / 2.0f;
    m_optimizer->setLearningRate(newLR);
}

} // namespace nn
} // namespace mcgan