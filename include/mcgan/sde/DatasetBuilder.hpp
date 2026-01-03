#pragma once

#include "SdeModel.hpp"
#include "DiscreteScheme.hpp"
#include "ExactSimulator.hpp"
#include "../nn/Tensor.hpp"
#include <vector>
#include <memory>
#include <string>

namespace mcgan {
namespace sde {

/**
 * Dataset containing SDE simulation data for GAN training.
 */
struct Dataset {
    std::vector<nn::Tensor> samples;      // Individual samples
    std::vector<nn::Tensor> conditions;   // Conditional information (if applicable)
    std::vector<float> labels;            // Labels for supervised learning
    std::vector<Path> paths;              // Full paths (optional)
    
    int numSamples() const { return samples.size(); }
    int getDimension() const { return samples.empty() ? 0 : samples[0].size(); }
    
    void clear() {
        samples.clear();
        conditions.clear();
        labels.clear();
        paths.clear();
    }
};

/**
 * Builder class for creating training datasets from SDE simulations.
 */
class DatasetBuilder {
public:
    DatasetBuilder(std::shared_ptr<SdeModel> model)
        : m_model(model)
        , m_useExactSimulation(false)
        , m_scheme(nullptr)
        , m_exactSimulator(nullptr)
    {}
    
    /**
     * Set simulation scheme (for numerical integration).
     */
    void setScheme(std::shared_ptr<DiscreteScheme> scheme) {
        m_scheme = scheme;
        m_useExactSimulation = false;
    }
    
    /**
     * Use exact simulation (if available).
     */
    void useExactSimulation(int seed = 0) {
        m_exactSimulator = std::make_shared<ExactSimulator>(m_model, seed);
        m_useExactSimulation = true;
    }
    
    /**
     * Build dataset of terminal values.
     * @param initialStates Initial conditions
     * @param T Terminal time
     * @param numSamplesPerInit Number of samples per initial condition
     * @return Dataset of terminal values
     */
    Dataset buildTerminalDataset(const std::vector<std::vector<float>>& initialStates,
                                 float T, int numSamplesPerInit) {
        Dataset dataset;
        
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
    
    /**
     * Build dataset of complete paths.
     * @param initialStates Initial conditions
     * @param T Terminal time
     * @param numSteps Number of time steps
     * @param numSamplesPerInit Number of samples per initial condition
     * @return Dataset with full paths
     */
    Dataset buildPathDataset(const std::vector<std::vector<float>>& initialStates,
                            float T, int numSteps, int numSamplesPerInit) {
        Dataset dataset;
        
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
    
    /**
     * Build dataset of path increments (for training dynamics).
     */
    Dataset buildIncrementDataset(const std::vector<std::vector<float>>& initialStates,
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
                    for (size_t d = 0; d < path.states[t].size(); d++) {
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
    
    /**
     * Build dataset with specific statistics (for variance reduction).
     */
    Dataset buildStatisticsDataset(const std::vector<std::vector<float>>& initialStates,
                                   float T, int numSteps, int numSamplesPerInit,
                                   const std::string& statistic = "mean") {
        Dataset dataset;
        
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
    
    /**
     * Split dataset into training and validation sets.
     */
    std::pair<Dataset, Dataset> splitDataset(const Dataset& data, float trainRatio = 0.8f) {
        int numTrain = static_cast<int>(data.numSamples() * trainRatio);
        
        Dataset train, val;
        
        for (int i = 0; i < data.numSamples(); i++) {
            if (i < numTrain) {
                train.samples.push_back(data.samples[i]);
                if (i < data.conditions.size()) {
                    train.conditions.push_back(data.conditions[i]);
                }
                if (i < data.labels.size()) {
                    train.labels.push_back(data.labels[i]);
                }
            } else {
                val.samples.push_back(data.samples[i]);
                if (i < data.conditions.size()) {
                    val.conditions.push_back(data.conditions[i]);
                }
                if (i < data.labels.size()) {
                    val.labels.push_back(data.labels[i]);
                }
            }
        }
        
        return {train, val};
    }
    
    /**
     * Normalize dataset to zero mean and unit variance.
     */
    void normalizeDataset(Dataset& dataset) {
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
        for (int i = 0; i < dim; i++) {
            mean[i] /= dataset.numSamples();
        }
        
        // Compute std
        for (const auto& sample : dataset.samples) {
            for (int i = 0; i < dim; i++) {
                float diff = sample[i] - mean[i];
                std[i] += diff * diff;
            }
        }
        for (int i = 0; i < dim; i++) {
            std[i] = std::sqrt(std[i] / dataset.numSamples());
            if (std[i] < 1e-8f) std[i] = 1.0f;  // Avoid division by zero
        }
        
        // Normalize
        for (auto& sample : dataset.samples) {
            for (int i = 0; i < dim; i++) {
                sample[i] = (sample[i] - mean[i]) / std[i];
            }
        }
        
        // Store normalization parameters
        m_normMean = mean;
        m_normStd = std;
    }
    
    /**
     * Get normalization parameters.
     */
    std::pair<std::vector<float>, std::vector<float>> getNormalizationParams() const {
        return {m_normMean, m_normStd};
    }

private:
    std::shared_ptr<SdeModel> m_model;
    std::shared_ptr<DiscreteScheme> m_scheme;
    std::shared_ptr<ExactSimulator> m_exactSimulator;
    bool m_useExactSimulation;
    
    std::vector<float> m_normMean;
    std::vector<float> m_normStd;
    
    std::vector<float> computePathStatistic(const Path& path, const std::string& statistic) {
        int dim = path.getDimension();
        std::vector<float> result(dim, 0.0f);
        
        if (statistic == "mean") {
            for (const auto& state : path.states) {
                for (int i = 0; i < dim; i++) {
                    result[i] += state[i];
                }
            }
            for (int i = 0; i < dim; i++) {
                result[i] /= path.states.size();
            }
        } else if (statistic == "terminal") {
            result = path.states.back();
        } else if (statistic == "max") {
            result = path.states[0];
            for (const auto& state : path.states) {
                for (int i = 0; i < dim; i++) {
                    result[i] = std::max(result[i], state[i]);
                }
            }
        } else if (statistic == "min") {
            result = path.states[0];
            for (const auto& state : path.states) {
                for (int i = 0; i < dim; i++) {
                    result[i] = std::min(result[i], state[i]);
                }
            }
        }
        
        return result;
    }
};

} // namespace sde
} // namespace mcgan