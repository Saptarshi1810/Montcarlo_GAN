#pragma once

#include "Tensor.hpp"
#include <memory>
#include <vector>

namespace mcgan {
namespace nn {

/**
 * Interface for GAN discriminator networks.
 * Discriminators classify samples as real or fake.
 */
class IDiscriminator {
public:
    virtual ~IDiscriminator() = default;
    
    /**
     * Discriminate samples (classify as real or fake).
     * @param samples Input samples to classify
     * @return Probabilities that samples are real (0 to 1)
     */
    virtual Tensor discriminate(const Tensor& samples) = 0;
    
    /**
     * Forward pass (alias for discriminate).
     */
    virtual Tensor forward(const Tensor& input) = 0;
    
    /**
     * Backward pass for training.
     * @param outputGrad Gradient from loss function
     * @return Gradient with respect to input
     */
    virtual Tensor backward(const Tensor& outputGrad) = 0;
    
    /**
     * Get discriminator parameters for optimization.
     */
    virtual std::vector<Tensor*> getParameters() = 0;
    
    /**
     * Get parameter gradients.
     */
    virtual std::vector<Tensor*> getGradients() = 0;
    
    /**
     * Set training mode.
     */
    virtual void setTraining(bool training) = 0;
    
    /**
     * Check if in training mode.
     */
    virtual bool isTraining() const = 0;
    
    /**
     * Get input dimension.
     */
    virtual int getInputDim() const = 0;
    
    /**
     * Initialize parameters.
     */
    virtual void initializeParameters(int seed = 0) = 0;
    
    /**
     * Save model to file.
     */
    virtual void save(const std::string& path) const = 0;
    
    /**
     * Load model from file.
     */
    virtual void load(const std::string& path) = 0;
    
    /**
     * Zero out gradients.
     */
    virtual void zeroGrad() = 0;
    
    /**
     * Get number of parameters.
     */
    virtual int getNumParameters() const = 0;
    
    /**
     * Clone the discriminator.
     */
    virtual std::unique_ptr<IDiscriminator> clone() const = 0;
    
    /**
     * Compute accuracy on a batch.
     * @param samples Input samples
     * @param labels True labels (1 for real, 0 for fake)
     * @return Accuracy (0 to 1)
     */
    virtual float computeAccuracy(const Tensor& samples, const Tensor& labels) = 0;
};

} // namespace nn
} // namespace mcgan