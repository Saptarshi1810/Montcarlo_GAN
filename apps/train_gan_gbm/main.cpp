#include <iostream>
#include <vector>
#include <torch/torch.h>
#include "../include/mcgan/core/Config.hpp"
#include "../include/mcgan/core/Logger.hpp"
#include "../include/mcgan/mc/ClassicalSampler.hpp"
#include "../include/mcgan/nn/Generator.hpp"
#include "../include/mcgan/nn/Discriminator.hpp"

// GAN training application for Geometric Brownian Motion paths
class GANTrainer {
private:
    Config config;
    Logger logger;
    torch::Device device;

    std::shared_ptr<Generator> generator;
    std::shared_ptr<Discriminator> discriminator;
    
    torch::optim::Adam* gen_optimizer;
    torch::optim::Adam* disc_optimizer;

    double S0, mu, sigma, T;
    int timeSteps;
    int latentDim;

public:
    GANTrainer(const std::string& configFile) 
        : config(configFile), 
          logger("GANTrainer"),
          device(torch::kCPU) {
        
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            logger.info("Using CUDA device");
        }

        // Load parameters from config
        S0 = config.getDouble("S0", 100.0);
        mu = config.getDouble("mu", 0.05);
        sigma = config.getDouble("sigma", 0.2);
        T = config.getDouble("T", 1.0);
        timeSteps = config.getInt("time_steps", 100);
        latentDim = config.getInt("latent_dim", 64);

        int hiddenDim = config.getInt("hidden_dim", 256);
        double learnRate = config.getDouble("learning_rate", 0.0002);

        // Initialize networks
        generator = std::make_shared<Generator>(latentDim, timeSteps, hiddenDim);
        discriminator = std::make_shared<Discriminator>(timeSteps, hiddenDim);

        generator->to(device);
        discriminator->to(device);

        // Initialize optimizers
        gen_optimizer = new torch::optim::Adam(
            generator->parameters(), 
            torch::optim::AdamOptions(learnRate).betas({0.5, 0.999}));
        
        disc_optimizer = new torch::optim::Adam(
            discriminator->parameters(), 
            torch::optim::AdamOptions(learnRate).betas({0.5, 0.999}));

        logger.info("GAN initialized with " + std::to_string(timeSteps) + " time steps");
    }

    ~GANTrainer() {
        delete gen_optimizer;
        delete disc_optimizer;
    }

    torch::Tensor generateRealPaths(int batchSize) {
        ClassicalSampler sampler(S0, mu, sigma, T, timeSteps);
        
        std::vector<std::vector<double>> paths;
        for (int i = 0; i < batchSize; ++i) {
            paths.push_back(sampler.generatePath());
        }

        // Convert to tensor
        torch::Tensor pathsTensor = torch::zeros({batchSize, timeSteps});
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < timeSteps; ++j) {
                pathsTensor[i][j] = paths[i][j];
            }
        }

        return pathsTensor.to(device);
    }

    torch::Tensor generateLatentVector(int batchSize) {
        return torch::randn({batchSize, latentDim}).to(device);
    }

    void trainDiscriminator(int batchSize) {
        disc_optimizer->zero_grad();

        // Real paths
        torch::Tensor realPaths = generateRealPaths(batchSize);
        torch::Tensor realLabels = torch::ones({batchSize, 1}).to(device);
        torch::Tensor realOutput = discriminator->forward(realPaths);
        torch::Tensor realLoss = torch::binary_cross_entropy(realOutput, realLabels);

        // Fake paths
        torch::Tensor noise = generateLatentVector(batchSize);
        torch::Tensor fakePaths = generator->forward(noise).detach();
        torch::Tensor fakeLabels = torch::zeros({batchSize, 1}).to(device);
        torch::Tensor fakeOutput = discriminator->forward(fakePaths);
        torch::Tensor fakeLoss = torch::binary_cross_entropy(fakeOutput, fakeLabels);

        torch::Tensor discLoss = realLoss + fakeLoss;
        discLoss.backward();
        disc_optimizer->step();
    }

    void trainGenerator(int batchSize) {
        gen_optimizer->zero_grad();

        torch::Tensor noise = generateLatentVector(batchSize);
        torch::Tensor fakePaths = generator->forward(noise);
        torch::Tensor fakeOutput = discriminator->forward(fakePaths);
        torch::Tensor realLabels = torch::ones({batchSize, 1}).to(device);
        
        torch::Tensor genLoss = torch::binary_cross_entropy(fakeOutput, realLabels);
        genLoss.backward();
        gen_optimizer->step();
    }

    void train(int epochs, int batchSize, int discSteps = 1) {
        logger.info("Starting GAN training for " + std::to_string(epochs) + " epochs");

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Train discriminator
            for (int k = 0; k < discSteps; ++k) {
                trainDiscriminator(batchSize);
            }

            // Train generator
            trainGenerator(batchSize);

            // Log progress
            if ((epoch + 1) % 100 == 0) {
                torch::NoGradGuard no_grad;
                
                torch::Tensor noise = generateLatentVector(batchSize);
                torch::Tensor fakePaths = generator->forward(noise);
                torch::Tensor fakeOutput = discriminator->forward(fakePaths);
                
                torch::Tensor realPaths = generateRealPaths(batchSize);
                torch::Tensor realOutput = discriminator->forward(realPaths);

                double avgFakeScore = fakeOutput.mean().item<double>();
                double avgRealScore = realOutput.mean().item<double>();

                logger.info("Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs) +
                           " - D(real)=" + std::to_string(avgRealScore) +
                           ", D(fake)=" + std::to_string(avgFakeScore));
            }

            // Save checkpoint
            if ((epoch + 1) % 1000 == 0) {
                saveModel("checkpoint_epoch_" + std::to_string(epoch + 1) + ".pt");
            }
        }

        saveModel("trained_gan_model.pt");
        logger.info("Training completed!");
    }

    void saveModel(const std::string& filename) {
        torch::save(generator, filename);
        logger.info("Model saved to " + filename);
    }

    void loadModel(const std::string& filename) {
        torch::load(generator, filename);
        logger.info("Model loaded from " + filename);
    }
};

int main(int argc, char* argv[]) {
    std::string configFile = "configs/default.yaml";
    if (argc > 1) {
        configFile = argv[1];
    }

    try {
        GANTrainer trainer(configFile);
        
        int epochs = 5000;
        int batchSize = 64;
        int discSteps = 1;

        if (argc > 2) epochs = std::stoi(argv[2]);
        if (argc > 3) batchSize = std::stoi(argv[3]);

        trainer.train(epochs, batchSize, discSteps);
        
        std::cout << "GAN training completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}