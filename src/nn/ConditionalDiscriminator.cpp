#include "ConditionalDiscriminator.hpp"
#include "ConditionalGenerator.hpp"  // For Layer definition
#include <fstream>
#include <iostream>

namespace mcgan {
namespace nn {

// ============================================================================
// ConditionalDiscriminator Implementation
// ============================================================================

ConditionalDiscriminator::ConditionalDiscriminator(
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

Tensor ConditionalDiscriminator::discriminate(const Tensor& samples) {
    // Discriminate without condition (use zero condition)
    Tensor condition({samples.shape()[0], m_conditionDim}, 0.0f);
    return discriminateConditional(samples, condition);
}

Tensor ConditionalDiscriminator::discriminateConditional(
    const Tensor& samples, const Tensor& condition) {
    
    if (samples.shape()[0] != condition.shape()[0]) {
        throw std::runtime_error("Batch sizes of samples and condition must match");
    }
    
    int batchSize = samples.shape()[0];
    Tensor concatenated({batchSize, m_inputDim + m_conditionDim});
    
    // Concatenate samples and condition
    for (int b = 0; b < batchSize; b++) {
        // Copy sample features
        for (int i = 0; i < m_inputDim; i++) {
            concatenated[b * (m_inputDim + m_conditionDim) + i] = 
                samples[b * m_inputDim + i];
        }
        // Copy condition features
        for (int i = 0; i < m_conditionDim; i++) {
            concatenated[b * (m_inputDim + m_conditionDim) + m_inputDim + i] = 
                condition[b * m_conditionDim + i];
        }
    }
    
    return forward(concatenated);
}

Tensor ConditionalDiscriminator::forward(const Tensor& input) {
    Tensor x = input;
    
    // Forward through hidden layers
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        Layer& layer = m_layers[i];
        
        // Linear transformation
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
    
    // Sigmoid activation for output (probability that sample is real)
    outputLayer.activation = applyActivation(outputLayer.preActivation, "sigmoid");
    
    return outputLayer.activation;
}

Tensor ConditionalDiscriminator::backward(const Tensor& outputGrad) {
    Tensor grad = outputGrad;
    
    // Backward through output layer
    Layer& outputLayer = m_layers.back();
    grad = applyActivationGradient(grad, outputLayer.activation, "sigmoid");
    
    // Compute gradients for output layer
    Tensor inputToOutput;
    if (m_layers.size() > 1) {
        inputToOutput = m_layers[m_layers.size() - 2].activation;
    } else {
        inputToOutput = Tensor({1, m_inputDim + m_conditionDim}, 0.0f);
    }
    
    outputLayer.weightsGrad = inputToOutput.transpose().matmul(grad);
    outputLayer.biasGrad = grad.sum(0, true);
    
    // Propagate gradient
    grad = grad.matmul(outputLayer.weights.transpose());
    
    // Backward through hidden layers
    for (int i = static_cast<int>(m_layers.size()) - 2; i >= 0; i--) {
        Layer& layer = m_layers[i];
        
        grad = applyActivationGradient(grad, layer.activation, "leaky_relu");
        
        Tensor inputToLayer;
        if (i > 0) {
            inputToLayer = m_layers[i - 1].activation;
        } else {
            inputToLayer = Tensor({1, m_inputDim + m_conditionDim}, 0.0f);
        }
        
        layer.weightsGrad = inputToLayer.transpose().matmul(grad);
        layer.biasGrad = grad.sum(0, true);
        
        if (i > 0) {
            grad = grad.matmul(layer.weights.transpose());
        }
    }
    
    return grad;
}

std::vector<Tensor*> ConditionalDiscriminator::getParameters() {
    std::vector<Tensor*> params;
    params.reserve(m_layers.size() * 2);
    
    for (Layer& layer : m_layers) {
        params.push_back(&layer.weights);
        params.push_back(&layer.bias);
    }
    
    return params;
}

std::vector<Tensor*> ConditionalDiscriminator::getGradients() {
    std::vector<Tensor*> grads;
    grads.reserve(m_layers.size() * 2);
    
    for (Layer& layer : m_layers) {
        grads.push_back(&layer.weightsGrad);
        grads.push_back(&layer.biasGrad);
    }
    
    return grads;
}

void ConditionalDiscriminator::initializeParameters(int seed) {
    for (size_t i = 0; i < m_layers.size(); i++) {
        m_layers[i].initializeHe(seed + static_cast<int>(i));
    }
}

void ConditionalDiscriminator::zeroGrad() {
    for (Layer& layer : m_layers) {
        layer.zeroGrad();
    }
}

int ConditionalDiscriminator::getNumParameters() const {
    int count = 0;
    for (const Layer& layer : m_layers) {
        count += layer.weights.size() + layer.bias.size();
    }
    return count;
}

void ConditionalDiscriminator::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    
    // Write architecture
    file.write(reinterpret_cast<const char*>(&m_inputDim), sizeof(m_inputDim));
    file.write(reinterpret_cast<const char*>(&m_conditionDim), sizeof(m_conditionDim));
    
    int numHiddenLayers = static_cast<int>(m_hiddenLayers.size());
    file.write(reinterpret_cast<const char*>(&numHiddenLayers), sizeof(numHiddenLayers));
    
    for (int dim : m_hiddenLayers) {
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    
    // Write parameters
    for (const Layer& layer : m_layers) {
        int weightSize = layer.weights.size();
        file.write(reinterpret_cast<const char*>(&weightSize), sizeof(weightSize));
        file.write(reinterpret_cast<const char*>(layer.weights.data()), 
                   weightSize * sizeof(float));
        
        int biasSize = layer.bias.size();
        file.write(reinterpret_cast<const char*>(&biasSize), sizeof(biasSize));
        file.write(reinterpret_cast<const char*>(layer.bias.data()), 
                   biasSize * sizeof(float));
    }
    
    file.close();
}

void ConditionalDiscriminator::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    
    // Read and validate architecture
    int inputDim, conditionDim;
    file.read(reinterpret_cast<char*>(&inputDim), sizeof(inputDim));
    file.read(reinterpret_cast<char*>(&conditionDim), sizeof(conditionDim));
    
    if (inputDim != m_inputDim || conditionDim != m_conditionDim) {
        throw std::runtime_error("Model architecture mismatch");
    }
    
    int numHiddenLayers;
    file.read(reinterpret_cast<char*>(&numHiddenLayers), sizeof(numHiddenLayers));
    
    for (int i = 0; i < numHiddenLayers; i++) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    }
    
    // Load parameters
    for (Layer& layer : m_layers) {
        int weightSize;
        file.read(reinterpret_cast<char*>(&weightSize), sizeof(weightSize));
        file.read(reinterpret_cast<char*>(layer.weights.data()), 
                  weightSize * sizeof(float));
        
        int biasSize;
        file.read(reinterpret_cast<char*>(&biasSize), sizeof(biasSize));
        file.read(reinterpret_cast<char*>(layer.bias.data()), 
                  biasSize * sizeof(float));
    }
    
    file.close();
}

std::unique_ptr<IDiscriminator> ConditionalDiscriminator::clone() const {
    auto cloned = std::make_unique<ConditionalDiscriminator>(
        m_inputDim, m_conditionDim, m_hiddenLayers);
    
    for (size_t i = 0; i < m_layers.size(); i++) {
        cloned->m_layers[i].weights = m_layers[i].weights;
        cloned->m_layers[i].bias = m_layers[i].bias;
    }
    
    cloned->m_dropoutRate = m_dropoutRate;
    
    return cloned;
}

float ConditionalDiscriminator::computeAccuracy(const Tensor& samples, const Tensor& labels) {
    Tensor predictions = discriminate(samples);
    
    int correct = 0;
    for (int i = 0; i < predictions.size(); i++) {
        float pred = predictions[i] > 0.5f ? 1.0f : 0.0f;
        if (std::abs(pred - labels[i]) < 0.1f) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / predictions.size();
}

Tensor ConditionalDiscriminator::applyActivation(const Tensor& input, const std::string& activation) {
    if (activation == "leaky_relu") {
        return input.leakyRelu(0.2f);
    } else if (activation == "sigmoid") {
        return input.sigmoid();
    } else if (activation == "tanh") {
        return input.tanh();
    } else if (activation == "relu") {
        return input.relu();
    }
    return input;
}

Tensor ConditionalDiscriminator::applyActivationGradient(
    const Tensor& grad, const Tensor& activation, const std::string& activationType) {
    
    if (activationType == "leaky_relu") {
        return grad * activation.leakyReluGradient(0.2f);
    } else if (activationType == "sigmoid") {
        return grad * activation.sigmoidGradient();
    } else if (activationType == "tanh") {
        return grad * activation.tanhGradient();
    } else if (activationType == "relu") {
        return grad * activation.reluGradient();
    }
    
    return grad;
}

Tensor ConditionalDiscriminator::dropout(const Tensor& input, float rate) {
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
            result[i] /= keepProb;  // Inverted dropout for inference
        }
    }
    
    return result;
}

} // namespace nn
} // namespace mcgan