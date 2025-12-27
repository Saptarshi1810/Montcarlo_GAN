#pragma once

#include "IDiscriminator.hpp"
#include "Tensor.hpp"
#include <vector>
#include <memory>

namespace mcgan {
namespace nn {

// Forward declare Layer from ConditionalGenerator
struct Layer;

/**
 * Conditional Discriminator that evaluates samples given a condition.
 * Used in conditional GANs to discriminate based on both sample and context.
 */
class ConditionalDiscriminator : public IDiscriminator {
public:
    ConditionalDiscriminator(int inputDim, int conditionDim,
                            const std::vector<int>& hiddenLayers);
    virtual ~ConditionalDiscriminator() = default;
    
    // IDiscriminator interface
    virtual Tensor discriminate(const Tensor& samples) override;
    virtual Tensor discriminateConditional(const Tensor& samples, const Tensor& condition);
    virtual Tensor forward(const Tensor& input) override;
    virtual Tensor backward(const Tensor& outputGrad) override;
    
    virtual std::vector<Tensor*> getParameters() override;
    virtual std::vector<Tensor*> getGradients() override;
    
    virtual void setTraining(bool training) override { m_training = training; }
    virtual bool isTraining() const override { return m_training; }
    
    virtual int getInputDim() const override { return m_inputDim; }
    int getConditionDim() const { return m_conditionDim; }
    
    virtual void initializeParameters(int seed = 0) override;
    virtual void save(const std::string& path) const override;
    virtual void load(const std::string& path) override;
    virtual void zeroGrad() override;
    virtual int getNumParameters() const override;
    virtual std::unique_ptr<IDiscriminator> clone() const override;
    
    virtual float computeAccuracy(const Tensor& samples, const Tensor& labels) override;
    
    // Additional methods
    void setDropoutRate(float rate) { m_dropoutRate = rate; }

protected:
    int m_inputDim;
    int m_conditionDim;
    std::vector<int> m_hiddenLayers;
    std::vector<Layer> m_layers;
    
    bool m_training;
    float m_dropoutRate;
    
    Tensor applyActivation(const Tensor& input, const std::string& activation);
    Tensor applyActivationGradient(const Tensor& grad, const Tensor& activation, 
                                   const std::string& activationType);
    Tensor dropout(const Tensor& input, float rate);
};

// ConditionalDiscriminator implementation
inline ConditionalDiscriminator::ConditionalDiscriminator(
    int inputDim, int conditionDim, const std::vector<int>& hiddenLayers)
    : m_inputDim(inputDim)
    , m_conditionDim(conditionDim)
    , m_hiddenLayers(hiddenLayers)
    , m_training(false)
    , m_dropoutRate(0.3f)
{
    // Build network architecture
    int totalInputDim = inputDim + conditionDim;
    int prevDim = totalInputDim;
    
    for (int hiddenDim : hiddenLayers) {
        m_layers.emplace_back(prevDim, hiddenDim, 0);
        prevDim = hiddenDim;
    }
    
    // Output layer (single output for binary classification)
    m_layers.emplace_back(prevDim, 1, 0);
}

inline Tensor ConditionalDiscriminator::discriminate(const Tensor& samples) {
    // Discriminate without condition (use zero condition)
    Tensor condition({samples.shape()[0], m_conditionDim}, 0.0f);
    return discriminateConditional(samples, condition);
}

inline Tensor ConditionalDiscriminator::discriminateConditional(
    const Tensor& samples, const Tensor& condition) {
    
    if (samples.shape()[0] != condition.shape()[0]) {
        throw std::runtime_error("Batch sizes of samples and condition must match");
    }
    
    int batchSize = samples.shape()[0];
    Tensor concatenated({batchSize, m_inputDim + m_conditionDim});
    
    // Concatenate samples and condition
    for (int b = 0; b < batchSize; b++) {
        for (int i = 0; i < m_inputDim; i++) {
            concatenated[b * (m_inputDim + m_conditionDim) + i] = samples[b * m_inputDim + i];
        }
        for (int i = 0; i < m_conditionDim; i++) {
            concatenated[b * (m_inputDim + m_conditionDim) + m_inputDim + i] = 
                condition[b * m_conditionDim + i];
        }
    }
    
    return forward(concatenated);
}

inline Tensor ConditionalDiscriminator::forward(const Tensor& input) {
    Tensor x = input;
    
    // Forward through hidden layers
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        Layer& layer = m_layers[i];
        
        // Linear transformation
        layer.preActivation = x.matmul(layer.weights);
        for (int j = 0; j < layer.preActivation.size(); j++) {
            layer.preActivation[j] += layer.bias[j % layer.bias.size()];
        }
        
        // Activation (LeakyReLU for hidden layers)
        layer.activation = applyActivation(layer.preActivation, "leaky_relu");
        
        // Dropout if training
        if (m_training && m_dropoutRate > 0.0f) {
            layer.activation = dropout(layer.activation, m_dropoutRate);
        }
        
        x = layer.activation;
    }
    
    // Output layer
    Layer& outputLayer = m_layers.back();
    outputLayer.preActivation = x.matmul(outputLayer.weights);
    for (int j = 0; j < outputLayer.preActivation.size(); j++) {
        outputLayer.preActivation[j] += outputLayer.bias[j % outputLayer.bias.size()];
    }
    
    // Sigmoid activation for output (probability)
    outputLayer.activation = applyActivation(outputLayer.preActivation, "sigmoid");
    
    return outputLayer.activation;
}

inline Tensor ConditionalDiscriminator::backward(const Tensor& outputGrad) {
    Tensor grad = outputGrad;
    
    // Backward through output layer
    Layer& outputLayer = m_layers.back();
    grad = applyActivationGradient(grad, outputLayer.activation, "sigmoid");
    
    // Compute gradients for output layer
    Tensor inputToOutput = m_layers.size() > 1 ? 
        m_layers[m_layers.size() - 2].activation : 
        Tensor({1, m_inputDim + m_conditionDim}, 0.0f);
    
    outputLayer.weightsGrad = inputToOutput.transpose().matmul(grad);
    outputLayer.biasGrad = grad.sum(0, true);
    
    // Propagate gradient
    grad = grad.matmul(outputLayer.weights.transpose());
    
    // Backward through hidden layers
    for (int i = m_layers.size() - 2; i >= 0; i--) {
        Layer& layer = m_layers[i];
        
        grad = applyActivationGradient(grad, layer.activation, "leaky_relu");
        
        Tensor inputToLayer = i > 0 ? 
            m_layers[i - 1].activation : 
            Tensor({1, m_inputDim + m_conditionDim}, 0.0f);
        
        layer.weightsGrad = inputToLayer.transpose().matmul(grad);
        layer.biasGrad = grad.sum(0, true);
        
        if (i > 0) {
            grad = grad.matmul(layer.weights.transpose());
        }
    }
    
    return grad;
}

inline std::vector<Tensor*> ConditionalDiscriminator::getParameters() {
    std::vector<Tensor*> params;
    for (Layer& layer : m_layers) {
        params.push_back(&layer.weights);
        params.push_back(&layer.bias);
    }
    return params;
}

inline std::vector<Tensor*> ConditionalDiscriminator::getGradients() {
    std::vector<Tensor*> grads;
    for (Layer& layer : m_layers) {
        grads.push_back(&layer.weightsGrad);
        grads.push_back(&layer.biasGrad);
    }
    return grads;
}

inline void ConditionalDiscriminator::initializeParameters(int seed) {
    for (size_t i = 0; i < m_layers.size(); i++) {
        m_layers[i].initializeHe(seed + i);
    }
}

inline void ConditionalDiscriminator::zeroGrad() {
    for (Layer& layer : m_layers) {
        layer.zeroGrad();
    }
}

inline int ConditionalDiscriminator::getNumParameters() const {
    int count = 0;
    for (const Layer& layer : m_layers) {
        count += layer.weights.size() + layer.bias.size();
    }
    return count;
}

inline void ConditionalDiscriminator::save(const std::string& path) const {
    // TODO: Implement serialization
}

inline void ConditionalDiscriminator::load(const std::string& path) {
    // TODO: Implement deserialization
}

inline std::unique_ptr<IDiscriminator> ConditionalDiscriminator::clone() const {
    auto cloned = std::make_unique<ConditionalDiscriminator>(
        m_inputDim, m_conditionDim, m_hiddenLayers);
    
    for (size_t i = 0; i < m_layers.size(); i++) {
        cloned->m_layers[i].weights = m_layers[i].weights;
        cloned->m_layers[i].bias = m_layers[i].bias;
    }
    
    return cloned;
}

inline float ConditionalDiscriminator::computeAccuracy(const Tensor& samples, const Tensor& labels) {
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

inline Tensor ConditionalDiscriminator::applyActivation(const Tensor& input, const std::string& activation) {
    if (activation == "leaky_relu") {
        return input.leakyRelu(0.2f);
    } else if (activation == "sigmoid") {
        return input.sigmoid();
    } else if (activation == "tanh") {
        return input.tanh();
    }
    return input;
}

inline Tensor ConditionalDiscriminator::applyActivationGradient(
    const Tensor& grad, const Tensor& activation, const std::string& activationType) {
    
    if (activationType == "leaky_relu") {
        return grad * activation.leakyReluGradient(0.2f);
    } else if (activationType == "sigmoid") {
        return grad * activation.sigmoidGradient();
    } else if (activationType == "tanh") {
        return grad * activation.tanhGradient();
    }
    return grad;
}

inline Tensor ConditionalDiscriminator::dropout(const Tensor& input, float rate) {
    Tensor mask = input;
    mask.randomUniform(0, 1);
    
    float keepProb = 1.0f - rate;
    Tensor result = input;
    
    for (int i = 0; i < input.size(); i++) {
        if (mask[i] > keepProb) {
            result[i] = 0.0f;
        } else {
            result[i] /= keepProb;
        }
    }
    
    return result;
}

} // namespace nn
} // namespace mcgan