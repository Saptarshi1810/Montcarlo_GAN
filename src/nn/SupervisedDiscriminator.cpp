#include "SupervisedDiscriminator.hpp"
#include "ConditionalGenerator.hpp"  // For Layer definition
#include <fstream>
#include <algorithm>

namespace mcgan {
namespace nn {

// ============================================================================
// SupervisedDiscriminator Implementation
// ============================================================================

SupervisedDiscriminator::SupervisedDiscriminator(
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

Tensor SupervisedDiscriminator::discriminate(const Tensor& samples) {
    m_sharedFeatures = forwardShared(samples);
    m_realFakeOutput = forwardRealFakeHead(m_sharedFeatures);
    return m_realFakeOutput;
}

Tensor SupervisedDiscriminator::classify(const Tensor& samples) {
    m_sharedFeatures = forwardShared(samples);
    m_classificationOutput = forwardClassificationHead(m_sharedFeatures);
    return m_classificationOutput;
}

Tensor SupervisedDiscriminator::classifyWithRealFakeScore(
    const Tensor& samples, Tensor& realFakeScore) {
    
    m_sharedFeatures = forwardShared(samples);
    realFakeScore = forwardRealFakeHead(m_sharedFeatures);
    m_realFakeOutput = realFakeScore;
    m_classificationOutput = forwardClassificationHead(m_sharedFeatures);
    return m_classificationOutput;
}

Tensor SupervisedDiscriminator::forward(const Tensor& input) {
    // Default forward returns real/fake discrimination
    return discriminate(input);
}

Tensor SupervisedDiscriminator::forwardShared(const Tensor& input) {
    Tensor x = input;
    
    for (size_t i = 0; i < m_sharedLayers.size(); i++) {
        Layer& layer = m_sharedLayers[i];
        
        // Linear transformation
        layer.preActivation = x.matmul(layer.weights);
        int outputSize = layer.preActivation.size();
        int biasSize = layer.bias.size();
        
        for (int j = 0; j < outputSize; j++) {
            layer.preActivation[j] += layer.bias[j % biasSize];
        }
        
        // LeakyReLU activation
        layer.activation = applyActivation(layer.preActivation, "leaky_relu");
        x = layer.activation;
    }
    
    return x;
}

Tensor SupervisedDiscriminator::forwardRealFakeHead(const Tensor& features) {
    Tensor x = features;
    
    for (size_t i = 0; i < m_realFakeLayers.size(); i++) {
        Layer& layer = m_realFakeLayers[i];
        
        layer.preActivation = x.matmul(layer.weights);
        int outputSize = layer.preActivation.size();
        int biasSize = layer.bias.size();
        
        for (int j = 0; j < outputSize; j++) {
            layer.preActivation[j] += layer.bias[j % biasSize];
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

Tensor SupervisedDiscriminator::forwardClassificationHead(const Tensor& features) {
    Tensor x = features;
    
    for (size_t i = 0; i < m_classificationLayers.size(); i++) {
        Layer& layer = m_classificationLayers[i];
        
        layer.preActivation = x.matmul(layer.weights);
        int outputSize = layer.preActivation.size();
        int biasSize = layer.bias.size();
        
        for (int j = 0; j < outputSize; j++) {
            layer.preActivation[j] += layer.bias[j % biasSize];
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

Tensor SupervisedDiscriminator::backward(const Tensor& outputGrad) {
    // Full backward pass through dual-head architecture
    Tensor grad = outputGrad;
    
    // Backward through real/fake head
    Layer& rfOutputLayer = m_realFakeLayers.back();
    Tensor rfGrad = applyActivationGradient(grad, rfOutputLayer.activation, "sigmoid");
    
    // Compute gradients for real/fake output layer
    Tensor inputToRFOutput = m_realFakeLayers.size() > 1 ? 
        m_realFakeLayers[m_realFakeLayers.size() - 2].activation : 
        m_sharedFeatures;
    
    rfOutputLayer.weightsGrad = inputToRFOutput.transpose().matmul(rfGrad);
    rfOutputLayer.biasGrad = rfGrad.sum(0, true);
    
    // Propagate gradient back
    Tensor gradToShared = rfGrad.matmul(rfOutputLayer.weights.transpose());
    
    // Backward through real/fake hidden layers
    for (int i = static_cast<int>(m_realFakeLayers.size()) - 2; i >= 0; i--) {
        Layer& layer = m_realFakeLayers[i];
        
        gradToShared = applyActivationGradient(gradToShared, layer.activation, "leaky_relu");
        
        Tensor inputToLayer = (i > 0) ? m_realFakeLayers[i - 1].activation : m_sharedFeatures;
        
        layer.weightsGrad = inputToLayer.transpose().matmul(gradToShared);
        layer.biasGrad = gradToShared.sum(0, true);
        
        if (i > 0) {
            gradToShared = gradToShared.matmul(layer.weights.transpose());
        }
    }
    
    // Backward through shared layers
    for (int i = static_cast<int>(m_sharedLayers.size()) - 1; i >= 0; i--) {
        Layer& layer = m_sharedLayers[i];
        
        gradToShared = applyActivationGradient(gradToShared, layer.activation, "leaky_relu");
        
        Tensor inputToLayer = (i > 0) ? m_sharedLayers[i - 1].activation : 
                              Tensor({1, m_inputDim}, 0.0f);
        
        layer.weightsGrad = inputToLayer.transpose().matmul(gradToShared);
        layer.biasGrad = gradToShared.sum(0, true);
        
        if (i > 0) {
            gradToShared = gradToShared.matmul(layer.weights.transpose());
        }
    }
    
    return gradToShared;
}

std::vector<Tensor*> SupervisedDiscriminator::getParameters() {
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

std::vector<Tensor*> SupervisedDiscriminator::getGradients() {
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

void SupervisedDiscriminator::initializeParameters(int seed) {
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

void SupervisedDiscriminator::zeroGrad() {
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

int SupervisedDiscriminator::getNumParameters() const {
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

void SupervisedDiscriminator::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    
    // Write architecture
    file.write(reinterpret_cast<const char*>(&m_inputDim), sizeof(m_inputDim));
    file.write(reinterpret_cast<const char*>(&m_numClasses), sizeof(m_numClasses));
    
    int numHiddenLayers = static_cast<int>(m_hiddenLayers.size());
    file.write(reinterpret_cast<const char*>(&numHiddenLayers), sizeof(numHiddenLayers));
    
    for (int dim : m_hiddenLayers) {
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    
    // Write shared layer parameters
    for (const Layer& layer : m_sharedLayers) {
        int weightSize = layer.weights.size();
        file.write(reinterpret_cast<const char*>(&weightSize), sizeof(weightSize));
        file.write(reinterpret_cast<const char*>(layer.weights.data()), 
                   weightSize * sizeof(float));
        
        int biasSize = layer.bias.size();
        file.write(reinterpret_cast<const char*>(&biasSize), sizeof(biasSize));
        file.write(reinterpret_cast<const char*>(layer.bias.data()), 
                   biasSize * sizeof(float));
    }
    
    // Write real/fake head parameters
    for (const Layer& layer : m_realFakeLayers) {
        int weightSize = layer.weights.size();
        file.write(reinterpret_cast<const char*>(&weightSize), sizeof(weightSize));
        file.write(reinterpret_cast<const char*>(layer.weights.data()), 
                   weightSize * sizeof(float));
        
        int biasSize = layer.bias.size();
        file.write(reinterpret_cast<const char*>(&biasSize), sizeof(biasSize));
        file.write(reinterpret_cast<const char*>(layer.bias.data()), 
                   biasSize * sizeof(float));
    }
    
    // Write classification head parameters
    for (const Layer& layer : m_classificationLayers) {
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

void SupervisedDiscriminator::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    
    // Read and validate architecture
    int inputDim, numClasses;
    file.read(reinterpret_cast<char*>(&inputDim), sizeof(inputDim));
    file.read(reinterpret_cast<char*>(&numClasses), sizeof(numClasses));
    
    if (inputDim != m_inputDim || numClasses != m_numClasses) {
        throw std::runtime_error("Model architecture mismatch");
    }
    
    int numHiddenLayers;
    file.read(reinterpret_cast<char*>(&numHiddenLayers), sizeof(numHiddenLayers));
    
    for (int i = 0; i < numHiddenLayers; i++) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    }
    
    // Load shared layer parameters
    for (Layer& layer : m_sharedLayers) {
        int weightSize;
        file.read(reinterpret_cast<char*>(&weightSize), sizeof(weightSize));
        file.read(reinterpret_cast<char*>(layer.weights.data()), 
                  weightSize * sizeof(float));
        
        int biasSize;
        file.read(reinterpret_cast<char*>(&biasSize), sizeof(biasSize));
        file.read(reinterpret_cast<char*>(layer.bias.data()), 
                  biasSize * sizeof(float));
    }
    
    // Load real/fake head parameters
    for (Layer& layer : m_realFakeLayers) {
        int weightSize;
        file.read(reinterpret_cast<char*>(&weightSize), sizeof(weightSize));
        file.read(reinterpret_cast<char*>(layer.weights.data()), 
                  weightSize * sizeof(float));
        
        int biasSize;
        file.read(reinterpret_cast<char*>(&biasSize), sizeof(biasSize));
        file.read(reinterpret_cast<char*>(layer.bias.data()), 
                  biasSize * sizeof(float));
    }
    
    // Load classification head parameters
    for (Layer& layer : m_classificationLayers) {
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

std::unique_ptr<IDiscriminator> SupervisedDiscriminator::clone() const {
    auto cloned = std::make_unique<SupervisedDiscriminator>(
        m_inputDim, m_numClasses, m_hiddenLayers);
    
    // Copy all layer parameters
    for (size_t i = 0; i < m_sharedLayers.size(); i++) {
        cloned->m_sharedLayers[i].weights = m_sharedLayers[i].weights;
        cloned->m_sharedLayers[i].bias = m_sharedLayers[i].bias;
    }
    
    for (size_t i = 0; i < m_realFakeLayers.size(); i++) {
        cloned->m_realFakeLayers[i].weights = m_realFakeLayers[i].weights;
        cloned->m_realFakeLayers[i].bias = m_realFakeLayers[i].bias;
    }
    
    for (size_t i = 0; i < m_classificationLayers.size(); i++) {
        cloned->m_classificationLayers[i].weights = m_classificationLayers[i].weights;
        cloned->m_classificationLayers[i].bias = m_classificationLayers[i].bias;
    }
    
    return cloned;
}

float SupervisedDiscriminator::computeAccuracy(const Tensor& samples, const Tensor& labels) {
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

float SupervisedDiscriminator::computeClassificationAccuracy(
    const Tensor& samples, const Tensor& labels) {
    
    Tensor predictions = classify(samples);
    
    int correct = 0;
    int batchSize = samples.shape()[0];
    
    for (int i = 0; i < batchSize; i++) {
        // Find argmax prediction
        int predClass = 0;
        float maxProb = predictions[i * m_numClasses];
        
        for (int c = 1; c < m_numClasses; c++) {
            if (predictions[i * m_numClasses + c] > maxProb) {
                maxProb = predictions[i * m_numClasses + c];
                predClass = c;
            }
        }
        
        int trueClass = static_cast<int>(labels[i]);
        if (predClass == trueClass) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / batchSize;
}

Tensor SupervisedDiscriminator::applyActivation(const Tensor& input, const std::string& activation) {
    if (activation == "leaky_relu") {
        return input.leakyRelu(0.2f);
    } else if (activation == "sigmoid") {
        return input.sigmoid();
    } else if (activation == "softmax") {
        return input.softmax(-1);
    } else if (activation == "relu") {
        return input.relu();
    } else if (activation == "tanh") {
        return input.tanh();
    }
    return input;
}

Tensor SupervisedDiscriminator::applyActivationGradient(
    const Tensor& grad, const Tensor& activation, const std::string& activationType) {
    
    if (activationType == "leaky_relu") {
        return grad * activation.leakyReluGradient(0.2f);
    } else if (activationType == "sigmoid") {
        return grad * activation.sigmoidGradient();
    } else if (activationType == "relu") {
        return grad * activation.reluGradient();
    } else if (activationType == "tanh") {
        return grad * activation.tanhGradient();
    }
    
    return grad;
}

} // namespace nn
} // namespace mcgan