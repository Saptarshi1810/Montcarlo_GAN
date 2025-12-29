#pragma once

#include "Tensor.hpp"
#include <memory>
#include <vector>

namespace mcgan {
namespace nn {

/**
 * Interface for GAN generator networks.
 * Generators transform latent vectors into samples.
 */
class IGenerator {
public:
    virtual ~IGenerator() = default;
    
    /**
     * Generate samples from latent vectors.
     * @param latent Input latent vectors (batch_size x latent_dim)
     * @return Generated samples
     */
    virtual Tensor generate(const Tensor& latent) = 0;
    
    /**
     * Forward pass (alias for generate).
     */
    virtual Tensor forward(const Tensor& input) = 0;
    
    /**
     * Backward pass for training.
     * @param outputGrad Gradient from discriminator
     * @return Gradient with respect to input
     */
    virtual Tensor backward(const Tensor& outputGrad) = 0;
    
    /**
     * Get generator parameters for optimization.
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
     * Get latent dimension.
     */
    virtual int getLatentDim() const = 0;
    
    /**
     * Get output dimension.
     */
    virtual int getOutputDim() const = 0;
    
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
     * Clone the generator.
     */
    virtual std::unique_ptr<IGenerator> clone() const = 0;
};

} // namespace nn
} // namespace mcgan