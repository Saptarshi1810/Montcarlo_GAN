#pragma once

#include "IGenerator.hpp"
#include "Tensor.hpp"
#include <vector>
#include <memory>

namespace mcgan {
namespace nn {

/**
 * Layer structure for neural network.
 */
struct Layer {
    Tensor weights;
    Tensor bias;
    Tensor weightsGrad;
    Tensor biasGrad;
    Tensor activation;
    Tensor preActivation;
    
    Layer() = default;
    Layer(int inputDim, int outputDim, int seed = 0);
    
    void initializeHe(int seed);
    void zeroGrad();
};

/**
 * Conditional Generator that takes both latent vector and condition.
 * Used for conditional GANs where generation depends on additional input.
 */
class ConditionalGenerator : public IGenerator {
public:
    ConditionalGenerator(int latentDim, int conditionDim, int outputDim,
                        const std::vector<int>& hiddenLayers);
    virtual ~ConditionalGenerator() = default;
    
    // IGenerator interface
    virtual Tensor generate(const Tensor& latent) override;
    virtual Tensor generateConditional(const Tensor& latent, const Tensor& condition);
    virtual Tensor forward(const Tensor& input) override;
    virtual Tensor backward(const Tensor& outputGrad) override;
    
    virtual std::vector<Tensor*> getParameters() override;
    virtual std::vector<Tensor*> getGradients() override;
    
    virtual void setTraining(bool training) override { m_training = training; }
    virtual bool isTraining() const override { return m_training; }
    
    virtual int getLatentDim() const override { return m_latentDim; }
    virtual int getOutputDim() const override { return m_outputDim; }
    int getConditionDim() const { return m_conditionDim; }
    
    virtual void initializeParameters(int seed = 0) override;
    virtual void save(const std::string& path) const override;
    virtual void load(const std::string& path) override;
    virtual void zeroGrad() override;
    virtual int getNumParameters() const override;
    virtual std::unique_ptr<IGenerator> clone() const override;
    
    // Additional methods
    void setDropoutRate(float rate) { m_dropoutRate = rate; }
    void setBatchNorm(bool enable) { m_useBatchNorm = enable; }

protected:
    int m_latentDim;
    int m_conditionDim;
    int m_outputDim;
    std::vector<int> m_hiddenLayers;
    std::vector<Layer> m_layers;
    
    bool m_training;
    float m_dropoutRate;
    bool m_useBatchNorm;
    
    Tensor applyActivation(const Tensor& input, const std::string& activation);
    Tensor applyActivationGradient(const Tensor& grad, const Tensor& activation, 
                                   const std::string& activationType);
    Tensor dropout(const Tensor& input, float rate);
};

// Layer implementation
inline Layer::Layer(int inputDim, int outputDim, int seed) 
    : weights({inputDim, outputDim})
    , bias({1, outputDim})
    , weightsGrad({inputDim, outputDim})
    , biasGrad({1, outputDim})
{
    initializeHe(seed);
    weightsGrad.zero();
    biasGrad.zero();
}

inline void Layer::initializeHe(int seed) {
    int fanIn = weights.shape()[0];
    weights.he(fanIn, seed);
    bias.zero();
}

inline void Layer::zeroGrad() {
    weightsGrad.zero();
    biasGrad.zero();
}

// ConditionalGenerator implementation
inline ConditionalGenerator::ConditionalGenerator(
    int latentDim, int conditionDim, int outputDim,
    const std::vector<int>& hiddenLayers)
    : m_latentDim(latentDim)
    , m_conditionDim(conditionDim)
    , m_outputDim(outputDim)
    , m_hiddenLayers(hiddenLayers)
    , m_training(false)
    , m_dropoutRate(0.0f)
    , m_useBatchNorm(false)
{
    // Build network architecture
    int inputDim = latentDim + conditionDim;
    int prevDim = inputDim;
    
    for (int hiddenDim : hiddenLayers) {
        m_layers.emplace_back(prevDim, hiddenDim, 0);
        prevDim = hiddenDim;
    }
    
    // Output layer
    m_layers.emplace_back(prevDim, outputDim, 0);
}

inline Tensor ConditionalGenerator::generate(const Tensor& latent) {
    // Generate without condition (use zero condition)
    Tensor condition({latent.shape()[0], m_conditionDim}, 0.0f);
    return generateConditional(latent, condition);
}

inline Tensor ConditionalGenerator::generateConditional(const Tensor& latent, const Tensor& condition) {
    // Concatenate latent and condition
    if (latent.shape()[0] != condition.shape()[0]) {
        throw std::runtime_error("Batch sizes of latent and condition must match");
    }
    
    int batchSize = latent.shape()[0];
    Tensor concatenated({batchSize, m_latentDim + m_conditionDim});
    
    // Copy latent
    for (int b = 0; b < batchSize; b++) {
        for (int i = 0; i < m_latentDim; i++) {
            concatenated[b * (m_latentDim + m_conditionDim) + i] = latent[b * m_latentDim + i];
        }
        for (int i = 0; i < m_conditionDim; i++) {
            concatenated[b * (m_latentDim + m_conditionDim) + m_latentDim + i] = 
                condition[b * m_conditionDim + i];
        }
    }
    
    return forward(concatenated);
}

inline Tensor ConditionalGenerator::forward(const Tensor& input) {
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
    
    // Tanh activation for output (scale to [-1, 1])
    outputLayer.activation = applyActivation(outputLayer.preActivation, "tanh");
    
    return outputLayer.activation;
}

inline Tensor ConditionalGenerator::backward(const Tensor& outputGrad) {
    Tensor grad = outputGrad;
    
    // Backward through output layer
    Layer& outputLayer = m_layers.back();
    grad = applyActivationGradient(grad, outputLayer.activation, "tanh");
    
    // Compute gradients for output layer
    Tensor inputToOutput = m_layers.size() > 1 ? 
        m_layers[m_layers.size() - 2].activation : 
        Tensor({1, m_latentDim + m_conditionDim}, 0.0f);
    
    outputLayer.weightsGrad = inputToOutput.transpose().matmul(grad);
    
    // Bias gradient (sum over batch)
    outputLayer.biasGrad = grad.sum(0, true);
    
    // Propagate gradient
    grad = grad.matmul(outputLayer.weights.transpose());
    
    // Backward through hidden layers
    for (int i = m_layers.size() - 2; i >= 0; i--) {
        Layer& layer = m_layers[i];
        
        grad = applyActivationGradient(grad, layer.activation, "leaky_relu");
        
        Tensor inputToLayer = i > 0 ? 
            m_layers[i - 1].activation : 
            Tensor({1, m_latentDim + m_conditionDim}, 0.0f);
        
        layer.weightsGrad = inputToLayer.transpose().matmul(grad);
        layer.biasGrad = grad.sum(0, true);
        
        if (i > 0) {
            grad = grad.matmul(layer.weights.transpose());
        }
    }
    
    return grad;
}

inline std::vector<Tensor*> ConditionalGenerator::getParameters() {
    std::vector<Tensor*> params;
    for (Layer& layer : m_layers) {
        params.push_back(&layer.weights);
        params.push_back(&layer.bias);
    }
    return params;
}

inline std::vector<Tensor*> ConditionalGenerator::getGradients() {
    std::vector<Tensor*> grads;
    for (Layer& layer : m_layers) {
        grads.push_back(&layer.weightsGrad);
        grads.push_back(&layer.biasGrad);
    }
    return grads;
}

inline void ConditionalGenerator::initializeParameters(int seed) {
    for (size_t i = 0; i < m_layers.size(); i++) {
        m_layers[i].initializeHe(seed + i);
    }
}

inline void ConditionalGenerator::zeroGrad() {
    for (Layer& layer : m_layers) {
        layer.zeroGrad();
    }
}

inline int ConditionalGenerator::getNumParameters() const {
    int count = 0;
    for (const Layer& layer : m_layers) {
        count += layer.weights.size() + layer.bias.size();
    }
    return count;
}

inline void ConditionalGenerator::save(const std::string& path) const {
    // TODO: Implement serialization
}

inline void ConditionalGenerator::load(const std::string& path) {
    // TODO: Implement deserialization
}

inline std::unique_ptr<IGenerator> ConditionalGenerator::clone() const {
    auto cloned = std::make_unique<ConditionalGenerator>(
        m_latentDim, m_conditionDim, m_outputDim, m_hiddenLayers);
    
    for (size_t i = 0; i < m_layers.size(); i++) {
        cloned->m_layers[i].weights = m_layers[i].weights;
        cloned->m_layers[i].bias = m_layers[i].bias;
    }
    
    return cloned;
}

inline Tensor ConditionalGenerator::applyActivation(const Tensor& input, const std::string& activation) {
    if (activation == "leaky_relu") {
        return input.leakyRelu(0.2f);
    } else if (activation == "tanh") {
        return input.tanh();
    } else if (activation == "sigmoid") {
        return input.sigmoid();
    }
    return input;
}

inline Tensor ConditionalGenerator::applyActivationGradient(
    const Tensor& grad, const Tensor& activation, const std::string& activationType) {
    
    if (activationType == "leaky_relu") {
        return grad * activation.leakyReluGradient(0.2f);
    } else if (activationType == "tanh") {
        return grad * activation.tanhGradient();
    } else if (activationType == "sigmoid") {
        return grad * activation.sigmoidGradient();
    }
    return grad;
}

inline Tensor ConditionalGenerator::dropout(const Tensor& input, float rate) {
    // Simple dropout implementation
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