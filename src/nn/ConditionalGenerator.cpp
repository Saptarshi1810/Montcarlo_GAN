#include "ConditionalGenerator.hpp"
#include <fstream>
#include <iostream>

namespace mcgan {
namespace nn {

// ============================================================================
// Layer Implementation
// ============================================================================

Layer::Layer(int inputDim, int outputDim, int seed) 
    : weights({inputDim, outputDim})
    , bias({1, outputDim})
    , weightsGrad({inputDim, outputDim})
    , biasGrad({1, outputDim})
{
    initializeHe(seed);
    weightsGrad.zero();
    biasGrad.zero();
}

void Layer::initializeHe(int seed) {
    int fanIn = weights.shape()[0];
    weights.he(fanIn, seed);
    bias.zero();
}

void Layer::zeroGrad() {
    weightsGrad.zero();
    biasGrad.zero();
}

// ============================================================================
// ConditionalGenerator Implementation
// ============================================================================

ConditionalGenerator::ConditionalGenerator(
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

Tensor ConditionalGenerator::generate(const Tensor& latent) {
    // Generate without condition (use zero condition)
    Tensor condition({latent.shape()[0], m_conditionDim}, 0.0f);
    return generateConditional(latent, condition);
}

Tensor ConditionalGenerator::generateConditional(const Tensor& latent, const Tensor& condition) {
    // Validate input dimensions
    if (latent.shape()[0] != condition.shape()[0]) {
        throw std::runtime_error("Batch sizes of latent and condition must match");
    }
    
    int batchSize = latent.shape()[0];
    Tensor concatenated({batchSize, m_latentDim + m_conditionDim});
    
    // Concatenate latent and condition vectors
    for (int b = 0; b < batchSize; b++) {
        // Copy latent vector
        for (int i = 0; i < m_latentDim; i++) {
            concatenated[b * (m_latentDim + m_conditionDim) + i] = 
                latent[b * m_latentDim + i];
        }
        // Copy condition vector
        for (int i = 0; i < m_conditionDim; i++) {
            concatenated[b * (m_latentDim + m_conditionDim) + m_latentDim + i] = 
                condition[b * m_conditionDim + i];
        }
    }
    
    return forward(concatenated);
}

Tensor ConditionalGenerator::forward(const Tensor& input) {
    Tensor x = input;
    
    // Forward through hidden layers
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        Layer& layer = m_layers[i];
        
        // Linear transformation: y = Wx + b
        layer.preActivation = x.matmul(layer.weights);
        int outputSize = layer.preActivation.size();
        int biasSize = layer.bias.size();
        
        for (int j = 0; j < outputSize; j++) {
            layer.preActivation[j] += layer.bias[j % biasSize];
        }
        
        // LeakyReLU activation for hidden layers
        layer.activation = applyActivation(layer.preActivation, "leaky_relu");
        
        // Apply dropout during training
        if (m_training && m_dropoutRate > 0.0f) {
            layer.activation = dropout(layer.activation, m_dropoutRate);
        }
        
        x = layer.activation;
    }
    
    // Output layer
    Layer& outputLayer = m_layers.back();
    outputLayer.preActivation = x.matmul(outputLayer.weights);
    
    int outputSize = outputLayer.preActivation.size();
    int biasSize = outputLayer.bias.size();
    
    for (int j = 0; j < outputSize; j++) {
        outputLayer.preActivation[j] += outputLayer.bias[j % biasSize];
    }
    
    // Tanh activation for output (scales to [-1, 1])
    outputLayer.activation = applyActivation(outputLayer.preActivation, "tanh");
    
    return outputLayer.activation;
}

Tensor ConditionalGenerator::backward(const Tensor& outputGrad) {
    Tensor grad = outputGrad;
    
    // Backward through output layer
    Layer& outputLayer = m_layers.back();
    grad = applyActivationGradient(grad, outputLayer.activation, "tanh");
    
    // Compute gradients for output layer weights and bias
    Tensor inputToOutput;
    if (m_layers.size() > 1) {
        inputToOutput = m_layers[m_layers.size() - 2].activation;
    } else {
        // First layer receives concatenated input
        inputToOutput = Tensor({1, m_latentDim + m_conditionDim}, 0.0f);
    }
    
    outputLayer.weightsGrad = inputToOutput.transpose().matmul(grad);
    outputLayer.biasGrad = grad.sum(0, true);
    
    // Propagate gradient to previous layer
    grad = grad.matmul(outputLayer.weights.transpose());
    
    // Backward through hidden layers
    for (int i = static_cast<int>(m_layers.size()) - 2; i >= 0; i--) {
        Layer& layer = m_layers[i];
        
        // Gradient through activation
        grad = applyActivationGradient(grad, layer.activation, "leaky_relu");
        
        // Input to this layer
        Tensor inputToLayer;
        if (i > 0) {
            inputToLayer = m_layers[i - 1].activation;
        } else {
            inputToLayer = Tensor({1, m_latentDim + m_conditionDim}, 0.0f);
        }
        
        // Compute weight and bias gradients
        layer.weightsGrad = inputToLayer.transpose().matmul(grad);
        layer.biasGrad = grad.sum(0, true);
        
        // Propagate gradient if not at first layer
        if (i > 0) {
            grad = grad.matmul(layer.weights.transpose());
        }
    }
    
    return grad;
}

std::vector<Tensor*> ConditionalGenerator::getParameters() {
    std::vector<Tensor*> params;
    params.reserve(m_layers.size() * 2);
    
    for (Layer& layer : m_layers) {
        params.push_back(&layer.weights);
        params.push_back(&layer.bias);
    }
    
    return params;
}

std::vector<Tensor*> ConditionalGenerator::getGradients() {
    std::vector<Tensor*> grads;
    grads.reserve(m_layers.size() * 2);
    
    for (Layer& layer : m_layers) {
        grads.push_back(&layer.weightsGrad);
        grads.push_back(&layer.biasGrad);
    }
    
    return grads;
}

void ConditionalGenerator::initializeParameters(int seed) {
    for (size_t i = 0; i < m_layers.size(); i++) {
        m_layers[i].initializeHe(seed + static_cast<int>(i));
    }
}

void ConditionalGenerator::zeroGrad() {
    for (Layer& layer : m_layers) {
        layer.zeroGrad();
    }
}

int ConditionalGenerator::getNumParameters() const {
    int count = 0;
    for (const Layer& layer : m_layers) {
        count += layer.weights.size() + layer.bias.size();
    }
    return count;
}

void ConditionalGenerator::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    
    // Write architecture information
    file.write(reinterpret_cast<const char*>(&m_latentDim), sizeof(m_latentDim));
    file.write(reinterpret_cast<const char*>(&m_conditionDim), sizeof(m_conditionDim));
    file.write(reinterpret_cast<const char*>(&m_outputDim), sizeof(m_outputDim));
    
    int numHiddenLayers = static_cast<int>(m_hiddenLayers.size());
    file.write(reinterpret_cast<const char*>(&numHiddenLayers), sizeof(numHiddenLayers));
    
    for (int dim : m_hiddenLayers) {
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    
    // Write layer parameters
    for (const Layer& layer : m_layers) {
        // Write weights
        int weightSize = layer.weights.size();
        file.write(reinterpret_cast<const char*>(&weightSize), sizeof(weightSize));
        file.write(reinterpret_cast<const char*>(layer.weights.data()), 
                   weightSize * sizeof(float));
        
        // Write bias
        int biasSize = layer.bias.size();
        file.write(reinterpret_cast<const char*>(&biasSize), sizeof(biasSize));
        file.write(reinterpret_cast<const char*>(layer.bias.data()), 
                   biasSize * sizeof(float));
    }
    
    file.close();
}

void ConditionalGenerator::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    
    // Read architecture (for validation)
    int latentDim, conditionDim, outputDim;
    file.read(reinterpret_cast<char*>(&latentDim), sizeof(latentDim));
    file.read(reinterpret_cast<char*>(&conditionDim), sizeof(conditionDim));
    file.read(reinterpret_cast<char*>(&outputDim), sizeof(outputDim));
    
    if (latentDim != m_latentDim || conditionDim != m_conditionDim || 
        outputDim != m_outputDim) {
        throw std::runtime_error("Model architecture mismatch");
    }
    
    int numHiddenLayers;
    file.read(reinterpret_cast<char*>(&numHiddenLayers), sizeof(numHiddenLayers));
    
    for (int i = 0; i < numHiddenLayers; i++) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    }
    
    // Load layer parameters
    for (Layer& layer : m_layers) {
        // Load weights
        int weightSize;
        file.read(reinterpret_cast<char*>(&weightSize), sizeof(weightSize));
        file.read(reinterpret_cast<char*>(layer.weights.data()), 
                  weightSize * sizeof(float));
        
        // Load bias
        int biasSize;
        file.read(reinterpret_cast<char*>(&biasSize), sizeof(biasSize));
        file.read(reinterpret_cast<char*>(layer.bias.data()), 
                  biasSize * sizeof(float));
    }
    
    file.close();
}

std::unique_ptr<IGenerator> ConditionalGenerator::clone() const {
    auto cloned = std::make_unique<ConditionalGenerator>(
        m_latentDim, m_conditionDim, m_outputDim, m_hiddenLayers);
    
    // Deep copy all layers
    for (size_t i = 0; i < m_layers.size(); i++) {
        cloned->m_layers[i].weights = m_layers[i].weights;
        cloned->m_layers[i].bias = m_layers[i].bias;
    }
    
    cloned->m_dropoutRate = m_dropoutRate;
    cloned->m_useBatchNorm = m_useBatchNorm;
    
    return cloned;
}

Tensor ConditionalGenerator::applyActivation(const Tensor& input, const std::string& activation) {
    if (activation == "leaky_relu") {
        return input.leakyRelu(0.2f);
    } else if (activation == "tanh") {
        return input.tanh();
    } else if (activation == "sigmoid") {
        return input.sigmoid();
    } else if (activation == "relu") {
        return input.relu();
    }
    return input;
}

Tensor ConditionalGenerator::applyActivationGradient(
    const Tensor& grad, const Tensor& activation, const std::string& activationType) {
    
    if (activationType == "leaky_relu") {
        return grad * activation.leakyReluGradient(0.2f);
    } else if (activationType == "tanh") {
        return grad * activation.tanhGradient();
    } else if (activationType == "sigmoid") {
        return grad * activation.sigmoidGradient();
    } else if (activationType == "relu") {
        return grad * activation.reluGradient();
    }
    
    return grad;
}

Tensor ConditionalGenerator::dropout(const Tensor& input, float rate) {
    if (rate <= 0.0f || rate >= 1.0f) {
        return input;
    }
    
    Tensor mask = input;
    mask.randomUniform(0.0f, 1.0f);
    
    float keepProb = 1.0f - rate;
    Tensor result = input;
    
    for (int i = 0; i < input.size(); i++) {
        if (mask[i] > keepProb) {
            result[i] = 0.0f;
        } else {
            result[i] /= keepProb;  // Inverted dropout
        }
    }
    
    return result;
}

} // namespace nn
} // namespace mcgan