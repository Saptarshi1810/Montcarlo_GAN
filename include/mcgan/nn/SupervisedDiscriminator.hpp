#pragma once

#include "IDiscriminator.hpp"
#include "Tensor.hpp"
#include <vector>
#include <memory>

namespace mcgan {
namespace nn {

// Forward declaration
struct Layer;

/**
 * Supervised Discriminator with auxiliary classification capabilities.
 * Can be trained with labeled data for semi-supervised learning.
 */
class SupervisedDiscriminator : public IDiscriminator {
public:
    SupervisedDiscriminator(int inputDim, int numClasses,
                           const std::vector<int>& hiddenLayers);
    virtual ~SupervisedDiscriminator() = default;
    
    // IDiscriminator interface
    virtual Tensor discriminate(const Tensor& samples) override;
    virtual Tensor forward(const Tensor& input) override;
    virtual Tensor backward(const Tensor& outputGrad) override;
    
    virtual std::vector<Tensor*> getParameters() override;
    virtual std::vector<Tensor*> getGradients() override;
    
    virtual void setTraining(bool training) override { m_training = training; }
    virtual bool isTraining() const override { return m_training; }
    
    virtual int getInputDim() const override { return m_inputDim; }
    
    virtual void initializeParameters(int seed = 0) override;
    virtual void save(const std::string& path) const override;
    virtual void load(const std::string& path) override;
    virtual void zeroGrad() override;
    virtual int getNumParameters() const override;
    virtual std::unique_ptr<IDiscriminator> clone() const override;
    
    virtual float computeAccuracy(const Tensor& samples, const Tensor& labels) override;
    
    // Supervised learning methods
    Tensor classify(const Tensor& samples);
    Tensor classifyWithRealFakeScore(const Tensor& samples, Tensor& realFakeScore);
    float computeClassificationAccuracy(const Tensor& samples, const Tensor& labels);
    
    // Dual-head architecture access
    Tensor getRealFakeOutput() const { return m_realFakeOutput; }
    Tensor getClassificationOutput() const { return m_classificationOutput; }
    
    int getNumClasses() const { return m_numClasses; }

protected:
    int m_inputDim;
    int m_numClasses;
    std::vector<int> m_hiddenLayers;
    std::vector<Layer> m_sharedLayers;
    
    // Dual heads
    std::vector<Layer> m_realFakeLayers;
    std::vector<Layer> m_classificationLayers;
    
    bool m_training;
    
    // Cached outputs
    Tensor m_realFakeOutput;
    Tensor m_classificationOutput;
    Tensor m_sharedFeatures;
    
    Tensor applyActivation(const Tensor& input, const std::string& activation);
    Tensor applyActivationGradient(const Tensor& grad, const Tensor& activation, 
                                   const std::string& activationType);
    
    Tensor forwardShared(const Tensor& input);
    Tensor forwardRealFakeHead(const Tensor& features);
    Tensor forwardClassificationHead(const Tensor& features);
};

// SupervisedDiscriminator implementation
inline SupervisedDiscriminator::SupervisedDiscriminator(
    int inputDim, int numClasses, const std::vector<int>& hiddenLayers)
    : m_inputDim(inputDim)
    , m_numClasses(numClasses)
    , m_hiddenLayers(hiddenLayers)
    , m_training(false)
{
    // Build shared feature extractor
    int prevDim = inputDim;
    for (int hiddenDim : hiddenLayers) {
        m_sharedLayers.emplace_back(prevDim, hiddenDim, 0);
        prevDim = hiddenDim;
    }
    
    // Real/Fake discrimination head (binary output)
    m_realFakeLayers.emplace_back(prevDim, 64, 0);
    m_realFakeLayers.emplace_back(64, 1, 0);
    
    // Classification head (multi-class output)
    m_classificationLayers.emplace_back(prevDim, 128, 0);
    m_classificationLayers.emplace_back(128, numClasses, 0);
}

inline Tensor SupervisedDiscriminator::discriminate(const Tensor& samples) {
    m_sharedFeatures = forwardShared(samples);
    m_realFakeOutput = forwardRealFakeHead(m_sharedFeatures);
    return m_realFakeOutput;
}

inline Tensor SupervisedDiscriminator::classify(const Tensor& samples) {
    m_sharedFeatures = forwardShared(samples);
    m_classificationOutput = forwardClassificationHead(m_sharedFeatures);
    return m_classificationOutput;
}

inline Tensor SupervisedDiscriminator::classifyWithRealFakeScore(
    const Tensor& samples, Tensor& realFakeScore) {
    
    m_sharedFeatures = forwardShared(samples);
    realFakeScore = forwardRealFakeHead(m_sharedFeatures);
    m_classificationOutput = forwardClassificationHead(m_sharedFeatures);
    return m_classificationOutput;
}

inline Tensor SupervisedDiscriminator::forward(const Tensor& input) {
    // Default forward returns real/fake discrimination
    return discriminate(input);
}

inline Tensor SupervisedDiscriminator::forwardShared(const Tensor& input) {
    Tensor x = input;
    
    for (size_t i = 0; i < m_sharedLayers.size(); i++) {
        Layer& layer = m_sharedLayers[i];
        
        layer.preActivation = x.matmul(layer.weights);
        for (int j = 0; j < layer.preActivation.size(); j++) {
            layer.preActivation[j] += layer.bias[j % layer.bias.size()];
        }
        
        layer.activation = applyActivation(layer.preActivation, "leaky_relu");
        x = layer.activation;
    }
    
    return x;
}

inline Tensor SupervisedDiscriminator::forwardRealFakeHead(const Tensor& features) {
    Tensor x = features;
    
    for (size_t i = 0; i < m_realFakeLayers.size(); i++) {
        Layer& layer = m_realFakeLayers[i];
        
        layer.preActivation = x.matmul(layer.weights);
        for (int j = 0; j < layer.preActivation.size(); j++) {
            layer.preActivation[j] += layer.bias[j % layer.bias.size()];
        }
        
        // Sigmoid on final layer, LeakyReLU on hidden layers
        if (i == m_realFakeLayers.size() - 1) {
            layer.activation = applyActivation(layer.preActivation, "sigmoid");
        } else {
            layer.activation = applyActivation(layer.preActivation, "leaky_relu");
        }
        
        x = layer.activation;
    }
    
    return x;
}

inline Tensor SupervisedDiscriminator::forwardClassificationHead(const Tensor& features) {
    Tensor x = features;
    
    for (size_t i = 0; i < m_classificationLayers.size(); i++) {
        Layer& layer = m_classificationLayers[i];
        
        layer.preActivation = x.matmul(layer.weights);
        for (int j = 0; j < layer.preActivation.size(); j++) {
            layer.preActivation[j] += layer.bias[j % layer.bias.size()];
        }
        
        // Softmax on final layer, LeakyReLU on hidden layers
        if (i == m_classificationLayers.size() - 1) {
            layer.activation = applyActivation(layer.preActivation, "softmax");
        } else {
            layer.activation = applyActivation(layer.preActivation, "leaky_relu");
        }
        
        x = layer.activation;
    }
    
    return x;
}

inline Tensor SupervisedDiscriminator::backward(const Tensor& outputGrad) {
    // Backward through real/fake head
    Tensor grad = outputGrad;
    
    // TODO: Implement full backward pass through dual-head architecture
    // This is a simplified version
    
    return grad;
}

inline std::vector<Tensor*> SupervisedDiscriminator::getParameters() {
    std::vector<Tensor*> params;
    
    // Shared layers
    for (Layer& layer : m_sharedLayers) {
        params.push_back(&layer.weights);
        params.push_back(&layer.bias);
    }
    
    // Real/Fake head
    for (Layer& layer : m_realFakeLayers) {
        params.push_back(&layer.weights);
        params.push_back(&layer.bias);
    }
    
    // Classification head
    for (Layer& layer : m_classificationLayers) {
        params.push_back(&layer.weights);
        params.push_back(&layer.bias);
    }
    
    return params;
}

inline std::vector<Tensor*> SupervisedDiscriminator::getGradients() {
    std::vector<Tensor*> grads;
    
    for (Layer& layer : m_sharedLayers) {
        grads.push_back(&layer.weightsGrad);
        grads.push_back(&layer.biasGrad);
    }
    
    for (Layer& layer : m_realFakeLayers) {
        grads.push_back(&layer.weightsGrad);
        grads.push_back(&layer.biasGrad);
    }
    
    for (Layer& layer : m_classificationLayers) {
        grads.push_back(&layer.weightsGrad);
        grads.push_back(&layer.biasGrad);
    }
    
    return grads;
}

inline void SupervisedDiscriminator::initializeParameters(int seed) {
    int currentSeed = seed;
    
    for (Layer& layer : m_sharedLayers) {
        layer.initializeHe(currentSeed++);
    }
    
    for (Layer& layer : m_realFakeLayers) {
        layer.initializeHe(currentSeed++);
    }
    
    for (Layer& layer : m_classificationLayers) {
        layer.initializeHe(currentSeed++);
    }
}

inline void SupervisedDiscriminator::zeroGrad() {
    for (Layer& layer : m_sharedLayers) {
        layer.zeroGrad();
    }
    
    for (Layer& layer : m_realFakeLayers) {
        layer.zeroGrad();
    }
    
    for (Layer& layer : m_classificationLayers) {
        layer.zeroGrad();
    }
}

inline int SupervisedDiscriminator::getNumParameters() const {
    int count = 0;
    
    for (const Layer& layer : m_sharedLayers) {
        count += layer.weights.size() + layer.bias.size();
    }
    
    for (const Layer& layer : m_realFakeLayers) {
        count += layer.weights.size() + layer.bias.size();
    }
    
    for (const Layer& layer : m_classificationLayers) {
        count += layer.weights.size() + layer.bias.size();
    }
    
    return count;
}

inline void SupervisedDiscriminator::save(const std::string& path) const {
    // TODO: Implement serialization
}

inline void SupervisedDiscriminator::load(const std::string& path) {
    // TODO: Implement deserialization
}

inline std::unique_ptr<IDiscriminator> SupervisedDiscriminator::clone() const {
    auto cloned = std::make_unique<SupervisedDiscriminator>(
        m_inputDim, m_numClasses, m_hiddenLayers);
    
    // Copy weights (simplified)
    // TODO: Implement full parameter copying
    
    return cloned;
}

inline float SupervisedDiscriminator::computeAccuracy(const Tensor& samples, const Tensor& labels) {
    Tensor predictions = discriminate(samples);
    
    int correct = 0;
    for (int i = 0; i < predictions.size(); i++) {
        float pred = predictions[i] > 0.5f ? 1.0f : 0.0f;
        if (pred == labels[i]) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / predictions.size();
}

inline float SupervisedDiscriminator::computeClassificationAccuracy(
    const Tensor& samples, const Tensor& labels) {
    
    Tensor predictions = classify(samples);
    
    int correct = 0;
    int batchSize = samples.shape()[0];
    
    for (int i = 0; i < batchSize; i++) {
        // Find argmax
        int predClass = 0;
        float maxProb = predictions[i * m_numClasses];
        
        for (int c = 1; c < m_numClasses; c++) {
            if (predictions[i * m_numClasses + c] > maxProb) {
                maxProb = predictions[i * m_numClasses + c];
                predClass = c;
            }
        }
        
        if (predClass == static_cast<int>(labels[i])) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / batchSize;
}

inline Tensor SupervisedDiscriminator::applyActivation(const Tensor& input, const std::string& activation) {
    if (activation == "leaky_relu") {
        return input.leakyRelu(0.2f);
    } else if (activation == "sigmoid") {
        return input.sigmoid();
    } else if (activation == "softmax") {
        return input.softmax(-1);
    }
    return input;
}

inline Tensor SupervisedDiscriminator::applyActivationGradient(
    const Tensor& grad, const Tensor& activation, const std::string& activationType) {
    
    if (activationType == "leaky_relu") {
        return grad * activation.leakyReluGradient(0.2f);
    } else if (activationType == "sigmoid") {
        return grad * activation.sigmoidGradient();
    }
    return grad;
}

} // namespace nn
} // namespace mcgan