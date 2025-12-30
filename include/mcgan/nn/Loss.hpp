#pragma once

#include "Tensor.hpp"
#include <cmath>
#include <algorithm>

namespace mcgan {
namespace nn {

/**
 * Loss function interface and implementations for GAN training.
 */
class Loss {
public:
    virtual ~Loss() = default;
    
    /**
     * Compute loss value.
     * @param predictions Model predictions
     * @param targets Target values
     * @return Loss value
     */
    virtual float compute(const Tensor& predictions, const Tensor& targets) = 0;
    
    /**
     * Compute gradient of loss with respect to predictions.
     * @param predictions Model predictions
     * @param targets Target values
     * @return Gradient tensor
     */
    virtual Tensor gradient(const Tensor& predictions, const Tensor& targets) = 0;
};

/**
 * Binary Cross-Entropy Loss for GAN discriminator.
 * L = -[y*log(p) + (1-y)*log(1-p)]
 */
class BCELoss : public Loss {
public:
    BCELoss(float epsilon = 1e-7f) : m_epsilon(epsilon) {}
    
    virtual float compute(const Tensor& predictions, const Tensor& targets) override {
        float loss = 0.0f;
        
        for (int i = 0; i < predictions.size(); i++) {
            float p = std::clamp(predictions[i], m_epsilon, 1.0f - m_epsilon);
            float y = targets[i];
            
            loss -= y * std::log(p) + (1.0f - y) * std::log(1.0f - p);
        }
        
        return loss / predictions.size();
    }
    
    virtual Tensor gradient(const Tensor& predictions, const Tensor& targets) override {
        Tensor grad(predictions.shape());
        
        for (int i = 0; i < predictions.size(); i++) {
            float p = std::clamp(predictions[i], m_epsilon, 1.0f - m_epsilon);
            float y = targets[i];
            
            grad[i] = (-y / p + (1.0f - y) / (1.0f - p)) / predictions.size();
        }
        
        return grad;
    }

private:
    float m_epsilon;
};

/**
 * Wasserstein Loss for WGAN.
 * L_D = E[D(fake)] - E[D(real)]
 * L_G = -E[D(fake)]
 */
class WassersteinLoss : public Loss {
public:
    WassersteinLoss() = default;
    
    virtual float compute(const Tensor& predictions, const Tensor& targets) override {
        // For discriminator: real samples (target=1), fake samples (target=-1)
        float loss = 0.0f;
        
        for (int i = 0; i < predictions.size(); i++) {
            // targets[i] = 1 for real, -1 for fake
            loss -= predictions[i] * targets[i];
        }
        
        return loss / predictions.size();
    }
    
    virtual Tensor gradient(const Tensor& predictions, const Tensor& targets) override {
        Tensor grad(predictions.shape());
        
        for (int i = 0; i < predictions.size(); i++) {
            grad[i] = -targets[i] / predictions.size();
        }
        
        return grad;
    }
};

/**
 * Least Squares GAN Loss (LSGAN).
 * L_D = 0.5 * E[(D(real) - 1)^2 + D(fake)^2]
 * L_G = 0.5 * E[(D(fake) - 1)^2]
 */
class LSGANLoss : public Loss {
public:
    LSGANLoss() = default;
    
    virtual float compute(const Tensor& predictions, const Tensor& targets) override {
        float loss = 0.0f;
        
        for (int i = 0; i < predictions.size(); i++) {
            float diff = predictions[i] - targets[i];
            loss += diff * diff;
        }
        
        return 0.5f * loss / predictions.size();
    }
    
    virtual Tensor gradient(const Tensor& predictions, const Tensor& targets) override {
        Tensor grad(predictions.shape());
        
        for (int i = 0; i < predictions.size(); i++) {
            grad[i] = (predictions[i] - targets[i]) / predictions.size();
        }
        
        return grad;
    }
};

/**
 * Hinge Loss for GANs.
 * L_D = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
 * L_G = -E[D(fake)]
 */
class HingeLoss : public Loss {
public:
    HingeLoss() = default;
    
    virtual float compute(const Tensor& predictions, const Tensor& targets) override {
        float loss = 0.0f;
        
        for (int i = 0; i < predictions.size(); i++) {
            if (targets[i] > 0.5f) {
                // Real samples: max(0, 1 - D(real))
                loss += std::max(0.0f, 1.0f - predictions[i]);
            } else {
                // Fake samples: max(0, 1 + D(fake))
                loss += std::max(0.0f, 1.0f + predictions[i]);
            }
        }
        
        return loss / predictions.size();
    }
    
    virtual Tensor gradient(const Tensor& predictions, const Tensor& targets) override {
        Tensor grad(predictions.shape());
        
        for (int i = 0; i < predictions.size(); i++) {
            if (targets[i] > 0.5f) {
                // Real samples
                grad[i] = (predictions[i] < 1.0f) ? -1.0f / predictions.size() : 0.0f;
            } else {
                // Fake samples
                grad[i] = (predictions[i] > -1.0f) ? 1.0f / predictions.size() : 0.0f;
            }
        }
        
        return grad;
    }
};

/**
 * Cross-Entropy Loss for classification tasks.
 */
class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss(float epsilon = 1e-7f) : m_epsilon(epsilon) {}
    
    virtual float compute(const Tensor& predictions, const Tensor& targets) override {
        // predictions: (batch_size, num_classes) - softmax probabilities
        // targets: (batch_size,) - class indices
        
        float loss = 0.0f;
        int batchSize = targets.size();
        int numClasses = predictions.size() / batchSize;
        
        for (int i = 0; i < batchSize; i++) {
            int targetClass = static_cast<int>(targets[i]);
            float p = std::clamp(predictions[i * numClasses + targetClass], m_epsilon, 1.0f);
            loss -= std::log(p);
        }
        
        return loss / batchSize;
    }
    
    virtual Tensor gradient(const Tensor& predictions, const Tensor& targets) override {
        Tensor grad(predictions.shape());
        grad.zero();
        
        int batchSize = targets.size();
        int numClasses = predictions.size() / batchSize;
        
        for (int i = 0; i < batchSize; i++) {
            int targetClass = static_cast<int>(targets[i]);
            
            // Gradient of cross-entropy with softmax
            for (int c = 0; c < numClasses; c++) {
                float p = predictions[i * numClasses + c];
                if (c == targetClass) {
                    grad[i * numClasses + c] = (p - 1.0f) / batchSize;
                } else {
                    grad[i * numClasses + c] = p / batchSize;
                }
            }
        }
        
        return grad;
    }

private:
    float m_epsilon;
};

/**
 * Mean Squared Error Loss.
 */
class MSELoss : public Loss {
public:
    MSELoss() = default;
    
    virtual float compute(const Tensor& predictions, const Tensor& targets) override {
        float loss = 0.0f;
        
        for (int i = 0; i < predictions.size(); i++) {
            float diff = predictions[i] - targets[i];
            loss += diff * diff;
        }
        
        return loss / predictions.size();
    }
    
    virtual Tensor gradient(const Tensor& predictions, const Tensor& targets) override {
        Tensor grad(predictions.shape());
        
        for (int i = 0; i < predictions.size(); i++) {
            grad[i] = 2.0f * (predictions[i] - targets[i]) / predictions.size();
        }
        
        return grad;
    }
};

/**
 * Gradient Penalty for WGAN-GP.
 * Enforces Lipschitz constraint on discriminator.
 */
class GradientPenalty {
public:
    GradientPenalty(float lambda = 10.0f) : m_lambda(lambda) {}
    
    /**
     * Compute gradient penalty.
     * @param discriminator The discriminator network
     * @param realSamples Real data samples
     * @param fakeSamples Generated samples
     * @return Gradient penalty value
     */
    float compute(const Tensor& gradients) {
        // Compute ||∇D(x)||_2
        float penalty = 0.0f;
        int batchSize = gradients.shape()[0];
        int dim = gradients.size() / batchSize;
        
        for (int i = 0; i < batchSize; i++) {
            float gradNorm = 0.0f;
            for (int j = 0; j < dim; j++) {
                float g = gradients[i * dim + j];
                gradNorm += g * g;
            }
            gradNorm = std::sqrt(gradNorm);
            
            // Penalty: (||∇D(x)||_2 - 1)^2
            float diff = gradNorm - 1.0f;
            penalty += diff * diff;
        }
        
        return m_lambda * penalty / batchSize;
    }
    
    float getLambda() const { return m_lambda; }
    void setLambda(float lambda) { m_lambda = lambda; }

private:
    float m_lambda;
};

/**
 * Feature Matching Loss for GANs.
 * Matches statistics of intermediate features between real and generated samples.
 */
class FeatureMatchingLoss : public Loss {
public:
    FeatureMatchingLoss() = default;
    
    virtual float compute(const Tensor& fakeFeatures, const Tensor& realFeatures) override {
        if (fakeFeatures.size() != realFeatures.size()) {
            throw std::runtime_error("Feature dimensions must match");
        }
        
        float loss = 0.0f;
        
        for (int i = 0; i < fakeFeatures.size(); i++) {
            float diff = fakeFeatures[i] - realFeatures[i];
            loss += diff * diff;
        }
        
        return std::sqrt(loss / fakeFeatures.size());
    }
    
    virtual Tensor gradient(const Tensor& fakeFeatures, const Tensor& realFeatures) override {
        Tensor grad(fakeFeatures.shape());
        
        float norm = 0.0f;
        for (int i = 0; i < fakeFeatures.size(); i++) {
            float diff = fakeFeatures[i] - realFeatures[i];
            norm += diff * diff;
        }
        norm = std::sqrt(norm);
        
        if (norm < 1e-8f) {
            grad.zero();
            return grad;
        }
        
        for (int i = 0; i < fakeFeatures.size(); i++) {
            grad[i] = (fakeFeatures[i] - realFeatures[i]) / (norm * fakeFeatures.size());
        }
        
        return grad;
    }
};

/**
 * Mode Seeking Loss to encourage diversity in generated samples.
 */
class ModeSeekingLoss : public Loss {
public:
    ModeSeekingLoss(float epsilon = 1e-5f) : m_epsilon(epsilon) {}
    
    virtual float compute(const Tensor& samples1, const Tensor& samples2) override {
        // Compute distance between samples in latent space and output space
        // Loss = dist(samples1, samples2) / (dist(latents1, latents2) + epsilon)
        
        float sampleDist = 0.0f;
        for (int i = 0; i < samples1.size(); i++) {
            float diff = samples1[i] - samples2[i];
            sampleDist += diff * diff;
        }
        sampleDist = std::sqrt(sampleDist);
        
        return -sampleDist;  // Negative to encourage larger distances
    }
    
    virtual Tensor gradient(const Tensor& samples1, const Tensor& samples2) override {
        Tensor grad(samples1.shape());
        
        float norm = 0.0f;
        for (int i = 0; i < samples1.size(); i++) {
            float diff = samples1[i] - samples2[i];
            norm += diff * diff;
        }
        norm = std::sqrt(norm);
        
        if (norm < m_epsilon) {
            grad.zero();
            return grad;
        }
        
        for (int i = 0; i < samples1.size(); i++) {
            grad[i] = -(samples1[i] - samples2[i]) / norm;
        }
        
        return grad;
    }

private:
    float m_epsilon;
};

} // namespace nn
} // namespace mcgan