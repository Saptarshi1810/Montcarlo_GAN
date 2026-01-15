#include "DatasetBuilder.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace mcgan {
namespace sde {

// ============================================================================
// Dataset Methods
// ============================================================================

int Dataset::numSamples() const {
    return static_cast<int>(samples.size());
}

int Dataset::getDimension() const {
    return samples.empty() ? 0 : samples[0].size();
}

void Dataset::clear() {
    samples.clear();
    conditions.clear();
    labels.clear();
    paths.clear();
}

// ============================================================================
// DatasetBuilder Implementation
// ============================================================================

Dataset DatasetBuilder::buildTerminalDataset(
    const std::vector<std::vector<float>>& initialStates,
    float T, int numSamplesPerInit) {
    
    Dataset dataset;
    dataset.samples.reserve(initialStates.size() * numSamplesPerInit);
    dataset.conditions.reserve(initialStates.size() * numSamplesPerInit);
    
    for (const auto& init : initialStates) {
        for (int i = 0; i < numSamplesPerInit; i++) {
            std::vector<float> terminal;
            
            if (m_useExactSimulation && m_exactSimulator) {
                terminal = m_exactSimulator->simulateTerminal(init, T);
            } else if (m_scheme) {
                int numSteps = 100;  // Default discretization
                Path path = m_scheme->simulate(init, T, numSteps);
                terminal = path.states.back();
            } else {
                throw std::runtime_error("No simulation method configured");
            }
            
            // Convert to tensor
            nn::Tensor sample(terminal, {static_cast<int>(terminal.size())});
            dataset.samples.push_back(sample);
            
            // Store initial condition as condition
            nn::Tensor condition(init, {static_cast<int>(init.size())});
            dataset.conditions.push_back(condition);
        }
    }
    
    return dataset;
}

Dataset DatasetBuilder::buildPathDataset(
    const std::vector<std::vector<float>>& initialStates,
    float T, int numSteps, int numSamplesPerInit) {
    
    Dataset dataset;
    dataset.samples.reserve(initialStates.size() * numSamplesPerInit);
    dataset.conditions.reserve(initialStates.size() * numSamplesPerInit);
    dataset.paths.reserve(initialStates.size() * numSamplesPerInit);
    
    for (const auto& init : initialStates) {
        for (int i = 0; i < numSamplesPerInit; i++) {
            Path path;
            
            if (m_useExactSimulation && m_exactSimulator) {
                path = m_exactSimulator->simulate(init, T, numSteps);
            } else if (m_scheme) {
                path = m_scheme->simulate(init, T, numSteps);
            } else {
                throw std::runtime_error("No simulation method configured");
            }
            
            // Flatten path to tensor
            std::vector<float> flatPath;
            int totalSize = static_cast<int>(path.states.size() * path.getDimension());
            flatPath.reserve(totalSize);
            
            for (const auto& state : path.states) {
                flatPath.insert(flatPath.end(), state.begin(), state.end());
            }
            
            nn::Tensor sample(flatPath, {static_cast<int>(flatPath.size())});
            dataset.samples.push_back(sample);
            dataset.paths.push_back(path);
            
            // Store initial condition
            nn::Tensor condition(init, {static_cast<int>(init.size())});
            dataset.conditions.push_back(condition);
        }
    }
    
    return dataset;
}

Dataset DatasetBuilder::buildIncrementDataset(
    const std::vector<std::vector<float>>& initialStates,
    float T, int numSteps, int numSamplesPerInit) {
    
    Dataset dataset;
    
    for (const auto& init : initialStates) {
        for (int i = 0; i < numSamplesPerInit; i++) {
            Path path;
            
            if (m_scheme) {
                path = m_scheme->simulate(init, T, numSteps);
            } else {
                throw std::runtime_error("Scheme required for increment dataset");
            }
            
            // Extract increments
            for (size_t t = 1; t < path.states.size(); t++) {
                std::vector<float> increment;
                int dim = static_cast<int>(path.states[t].size());
                increment.reserve(dim);
                
                for (int d = 0; d < dim; d++) {
                    increment.push_back(path.states[t][d] - path.states[t-1][d]);
                }
                
                nn::Tensor sample(increment, {static_cast<int>(increment.size())});
                dataset.samples.push_back(sample);
                
                // Condition on current state and time
                std::vector<float> conditionVec = path.states[t-1];
                conditionVec.push_back(path.times[t-1]);
                
                nn::Tensor condition(conditionVec, {static_cast<int>(conditionVec.size())});
                dataset.conditions.push_back(condition);
            }
        }
    }
    
    return dataset;
}

Dataset DatasetBuilder::buildStatisticsDataset(
    const std::vector<std::vector<float>>& initialStates,
    float T, int numSteps, int numSamplesPerInit,
    const std::string& statistic) {
    
    Dataset dataset;
    dataset.samples.reserve(initialStates.size() * numSamplesPerInit);
    dataset.conditions.reserve(initialStates.size() * numSamplesPerInit);
    
    for (const auto& init : initialStates) {
        for (int i = 0; i < numSamplesPerInit; i++) {
            Path path;
            
            if (m_useExactSimulation && m_exactSimulator) {
                path = m_exactSimulator->simulate(init, T, numSteps);
            } else if (m_scheme) {
                path = m_scheme->simulate(init, T, numSteps);
            } else {
                throw std::runtime_error("No simulation method configured");
            }
            
            // Compute statistic
            std::vector<float> stats = computePathStatistic(path, statistic);
            
            nn::Tensor sample(stats, {static_cast<int>(stats.size())});
            dataset.samples.push_back(sample);
            
            nn::Tensor condition(init, {static_cast<int>(init.size())});
            dataset.conditions.push_back(condition);
        }
    }
    
    return dataset;
}

std::pair<Dataset, Dataset> DatasetBuilder::splitDataset(const Dataset& data, float trainRatio) {
    int numTrain = static_cast<int>(data.numSamples() * trainRatio);
    
    Dataset train, val;
    
    train.samples.reserve(numTrain);
    val.samples.reserve(data.numSamples() - numTrain);
    
    for (int i = 0; i < data.numSamples(); i++) {
        if (i < numTrain) {
            train.samples.push_back(data.samples[i]);
            if (i < static_cast<int>(data.conditions.size())) {
                train.conditions.push_back(data.conditions[i]);
            }
            if (i < static_cast<int>(data.labels.size())) {
                train.labels.push_back(data.labels[i]);
            }
            if (i < static_cast<int>(data.paths.size())) {
                train.paths.push_back(data.paths[i]);
            }
        } else {
            val.samples.push_back(data.samples[i]);
            if (i < static_cast<int>(data.conditions.size())) {
                val.conditions.push_back(data.conditions[i]);
            }
            if (i < static_cast<int>(data.labels.size())) {
                val.labels.push_back(data.labels[i]);
            }
            if (i < static_cast<int>(data.paths.size())) {
                val.paths.push_back(data.paths[i]);
            }
        }
    }
    
    return {train, val};
}

void DatasetBuilder::normalizeDataset(Dataset& dataset) {
    if (dataset.samples.empty()) return;
    
    int dim = dataset.getDimension();
    std::vector<float> mean(dim, 0.0f);
    std::vector<float> std(dim, 0.0f);
    
    // Compute mean
    for (const auto& sample : dataset.samples) {
        for (int i = 0; i < dim; i++) {
            mean[i] += sample[i];
        }
    }
    
    int numSamples = dataset.numSamples();
    for (int i = 0; i < dim; i++) {
        mean[i] /= numSamples;
    }
    
    // Compute standard deviation
    for (const auto& sample : dataset.samples) {
        for (int i = 0; i < dim; i++) {
            float diff = sample[i] - mean[i];
            std[i] += diff * diff;
        }
    }
    
    for (int i = 0; i < dim; i++) {
        std[i] = std::sqrt(std[i] / numSamples);
        if (std[i] < 1e-8f) {
            std[i] = 1.0f;  // Avoid division by zero
        }
    }
    
    // Normalize samples
    for (auto& sample : dataset.samples) {
        for (int i = 0; i < dim; i++) {
            sample[i] = (sample[i] - mean[i]) / std[i];
        }
    }
    
    // Store normalization parameters
    m_normMean = mean;
    m_normStd = std;
}

std::pair<std::vector<float>, std::vector<float>> DatasetBuilder::getNormalizationParams() const {
    return {m_normMean, m_normStd};
}

std::vector<float> DatasetBuilder::computePathStatistic(const Path& path, const std::string& statistic) {
    int dim = path.getDimension();
    std::vector<float> result(dim, 0.0f);
    
    if (statistic == "mean") {
        // Compute time-average
        for (const auto& state : path.states) {
            for (int i = 0; i < dim; i++) {
                result[i] += state[i];
            }
        }
        
        int numSteps = static_cast<int>(path.states.size());
        for (int i = 0; i < dim; i++) {
            result[i] /= numSteps;
        }
    } 
    else if (statistic == "terminal") {
        // Terminal value
        result = path.states.back();
    } 
    else if (statistic == "max") {
        // Running maximum
        result = path.states[0];
        for (const auto& state : path.states) {
            for (int i = 0; i < dim; i++) {
                result[i] = std::max(result[i], state[i]);
            }
        }
    } 
    else if (statistic == "min") {
        // Running minimum
        result = path.states[0];
        for (const auto& state : path.states) {
            for (int i = 0; i < dim; i++) {
                result[i] = std::min(result[i], state[i]);
            }
        }
    }
    else if (statistic == "variance") {
        // Temporal variance
        std::vector<float> mean(dim, 0.0f);
        
        // First compute mean
        for (const auto& state : path.states) {
            for (int i = 0; i < dim; i++) {
                mean[i] += state[i];
            }
        }
        
        int numSteps = static_cast<int>(path.states.size());
        for (int i = 0; i < dim; i++) {
            mean[i] /= numSteps;
        }
        
        // Then compute variance
        for (const auto& state : path.states) {
            for (int i = 0; i < dim; i++) {
                float diff = state[i] - mean[i];
                result[i] += diff * diff;
            }
        }
        
        for (int i = 0; i < dim; i++) {
            result[i] /= numSteps;
        }
    }
    else {
        throw std::runtime_error("Unknown statistic: " + statistic);
    }
    
    return result;
}

} // namespace sde
} // namespace mcgan