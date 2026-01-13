#include "Loss.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace mcgan {
namespace nn {

// ============================================================================
// Binary Cross-Entropy Loss
// ============================================================================

float BCELoss::compute(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    float loss = 0.0f;
    
    for (int i = 0; i < predictions.size(); i++) {
        float p = std::clamp(predictions[i], m_epsilon, 1.0f - m_epsilon);
        float y = targets[i];
        
        loss -= y * std::log(p) + (1.0f - y) * std::log(1.0f - p);
    }
    
    return loss / predictions.size();
}

Tensor BCELoss::gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    Tensor grad(predictions.shape());
    
    for (int i = 0; i < predictions.size(); i++) {
        float p = std::clamp(predictions[i], m_epsilon, 1.0f - m_epsilon);
        float y = targets[i];
        
        // Gradient: -y/p + (1-y)/(1-p)
        grad[i] = (-y / p + (1.0f - y) / (1.0f - p)) / predictions.size();
    }
    
    return grad;
}

// ============================================================================
// Wasserstein Loss (WGAN)
// ============================================================================

float WassersteinLoss::compute(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    float loss = 0.0f;
    
    for (int i = 0; i < predictions.size(); i++) {
        // targets[i] = 1 for real, -1 for fake
        // Loss = -E[D(real)] + E[D(fake)] = -E[D(x) * target]
        loss -= predictions[i] * targets[i];
    }
    
    return loss / predictions.size();
}

Tensor WassersteinLoss::gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    Tensor grad(predictions.shape());
    
    for (int i = 0; i < predictions.size(); i++) {
        // Gradient: -target
        grad[i] = -targets[i] / predictions.size();
    }
    
    return grad;
}

// ============================================================================
// Least Squares GAN Loss
// ============================================================================

float LSGANLoss::compute(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    float loss = 0.0f;
    
    for (int i = 0; i < predictions.size(); i++) {
        float diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    
    return 0.5f * loss / predictions.size();
}

Tensor LSGANLoss::gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    Tensor grad(predictions.shape());
    
    for (int i = 0; i < predictions.size(); i++) {
        // Gradient: (prediction - target)
        grad[i] = (predictions[i] - targets[i]) / predictions.size();
    }
    
    return grad;
}

// ============================================================================
// Hinge Loss
// ============================================================================

float HingeLoss::compute(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
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

Tensor HingeLoss::gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    Tensor grad(predictions.shape());
    
    for (int i = 0; i < predictions.size(); i++) {
        if (targets[i] > 0.5f) {
            // Real samples: gradient is -1 if D(real) < 1, else 0
            grad[i] = (predictions[i] < 1.0f) ? -1.0f / predictions.size() : 0.0f;
        } else {
            // Fake samples: gradient is 1 if D(fake) > -1, else 0
            grad[i] = (predictions[i] > -1.0f) ? 1.0f / predictions.size() : 0.0f;
        }
    }
    
    return grad;
}

// ============================================================================
// Cross-Entropy Loss
// ============================================================================

float CrossEntropyLoss::compute(const Tensor& predictions, const Tensor& targets) {
    // predictions: (batch_size, num_classes) - softmax probabilities
    // targets: (batch_size,) - class indices or (batch_size, num_classes) - one-hot
    
    float loss = 0.0f;
    int batchSize = targets.size();
    
    if (predictions.size() == targets.size()) {
        // One-hot encoded targets
        for (int i = 0; i < predictions.size(); i++) {
            if (targets[i] > 0.5f) {
                float p = std::clamp(predictions[i], m_epsilon, 1.0f);
                loss -= std::log(p);
            }
        }
    } else {
        // Class indices
        int numClasses = predictions.size() / batchSize;
        
        for (int i = 0; i < batchSize; i++) {
            int targetClass = static_cast<int>(targets[i]);
            if (targetClass >= 0 && targetClass < numClasses) {
                float p = std::clamp(predictions[i * numClasses + targetClass], m_epsilon, 1.0f);
                loss -= std::log(p);
            }
        }
    }
    
    return loss / batchSize;
}

Tensor CrossEntropyLoss::gradient(const Tensor& predictions, const Tensor& targets) {
    Tensor grad(predictions.shape());
    grad.zero();
    
    int batchSize = targets.size();
    
    if (predictions.size() == targets.size()) {
        // One-hot encoded targets
        for (int i = 0; i < predictions.size(); i++) {
            float p = predictions[i];
            float y = targets[i];
            grad[i] = (p - y) / batchSize;
        }
    } else {
        // Class indices
        int numClasses = predictions.size() / batchSize;
        
        for (int i = 0; i < batchSize; i++) {
            int targetClass = static_cast<int>(targets[i]);
            
            if (targetClass >= 0 && targetClass < numClasses) {
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
        }
    }
    
    return grad;
}

// ============================================================================
// Mean Squared Error Loss
// ============================================================================

float MSELoss::compute(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    float loss = 0.0f;
    
    for (int i = 0; i < predictions.size(); i++) {
        float diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    
    return loss / predictions.size();
}

Tensor MSELoss::gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    Tensor grad(predictions.shape());
    
    for (int i = 0; i < predictions.size(); i++) {
        grad[i] = 2.0f * (predictions[i] - targets[i]) / predictions.size();
    }
    
    return grad;
}

// ============================================================================
// Gradient Penalty (WGAN-GP)
// ============================================================================

float GradientPenalty::compute(const Tensor& gradients) {
    // Compute ||∇D(x)||_2 for each sample in batch
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

// ============================================================================
// Feature Matching Loss
// ============================================================================

float FeatureMatchingLoss::compute(const Tensor& fakeFeatures, const Tensor& realFeatures) {
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

Tensor FeatureMatchingLoss::gradient(const Tensor& fakeFeatures, const Tensor& realFeatures) {
    if (fakeFeatures.size() != realFeatures.size()) {
        throw std::runtime_error("Feature dimensions must match");
    }
    
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

// ============================================================================
// Mode Seeking Loss
// ============================================================================

float ModeSeekingLoss::compute(const Tensor& samples1, const Tensor& samples2) {
    if (samples1.size() != samples2.size()) {
        throw std::runtime_error("Sample dimensions must match");
    }
    
    // Compute L2 distance between samples
    float sampleDist = 0.0f;
    for (int i = 0; i < samples1.size(); i++) {
        float diff = samples1[i] - samples2[i];
        sampleDist += diff * diff;
    }
    sampleDist = std::sqrt(sampleDist);
    
    // Return negative distance to encourage diversity
    return -sampleDist;
}

Tensor ModeSeekingLoss::gradient(const Tensor& samples1, const Tensor& samples2) {
    if (samples1.size() != samples2.size()) {
        throw std::runtime_error("Sample dimensions must match");
    }
    
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
    
    // Gradient of -||samples1 - samples2||_2
    for (int i = 0; i < samples1.size(); i++) {
        grad[i] = -(samples1[i] - samples2[i]) / norm;
    }
    
    return grad;
}

} // namespace nn
} // namespace mcgan