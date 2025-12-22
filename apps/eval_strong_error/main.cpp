#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include "../include/mcgan/core/Config.hpp"
#include "../include/mcgan/core/Logger.hpp"
#include "../include/mcgan/mc/ClassicalSampler.hpp"
#include "../include/mcgan/mc/GanPathSampler.hpp"

// Evaluate strong error convergence for GBM paths
class StrongErrorEvaluator {
private:
    Config config;
    Logger logger;
    std::mt19937 rng;

public:
    StrongErrorEvaluator(const std::string& configFile) 
        : config(configFile), logger("StrongErrorEval"), rng(std::random_device{}()) {}

    double computeStrongError(const std::vector<double>& truePath, 
                              const std::vector<double>& approxPath) {
        double sumSquaredError = 0.0;
        for (size_t i = 0; i < truePath.size(); ++i) {
            double diff = truePath[i] - approxPath[i];
            sumSquaredError += diff * diff;
        }
        return std::sqrt(sumSquaredError / truePath.size());
    }

    void evaluateClassicalVsGan(int numPaths, const std::vector<int>& timeSteps) {
        double S0 = config.getDouble("S0", 100.0);
        double mu = config.getDouble("mu", 0.05);
        double sigma = config.getDouble("sigma", 0.2);
        double T = config.getDouble("T", 1.0);

        std::ofstream outFile("strong_error_results.csv");
        outFile << "Method,TimeSteps,MeanError,StdError\n";

        for (int N : timeSteps) {
            logger.info("Evaluating with " + std::to_string(N) + " time steps");

            ClassicalSampler classicalSampler(S0, mu, sigma, T, N);
            GanPathSampler ganSampler("trained_gan_model.pt", S0, mu, sigma, T, N);

            std::vector<double> classicalErrors;
            std::vector<double> ganErrors;

            // Generate reference path with very fine discretization
            int refSteps = N * 10;
            ClassicalSampler refSampler(S0, mu, sigma, T, refSteps);

            for (int i = 0; i < numPaths; ++i) {
                auto refPath = refSampler.generatePath();
                
                // Downsample reference path
                std::vector<double> refPathCoarse;
                for (int j = 0; j < N; ++j) {
                    refPathCoarse.push_back(refPath[j * 10]);
                }

                auto classicalPath = classicalSampler.generatePath();
                auto ganPath = ganSampler.generatePath();

                classicalErrors.push_back(computeStrongError(refPathCoarse, classicalPath));
                ganErrors.push_back(computeStrongError(refPathCoarse, ganPath));
            }

            // Compute statistics
            double classicalMean = std::accumulate(classicalErrors.begin(), classicalErrors.end(), 0.0) / classicalErrors.size();
            double ganMean = std::accumulate(ganErrors.begin(), ganErrors.end(), 0.0) / ganErrors.size();

            double classicalStd = 0.0, ganStd = 0.0;
            for (const auto& err : classicalErrors) classicalStd += (err - classicalMean) * (err - classicalMean);
            for (const auto& err : ganErrors) ganStd += (err - ganMean) * (err - ganMean);
            classicalStd = std::sqrt(classicalStd / classicalErrors.size());
            ganStd = std::sqrt(ganStd / ganErrors.size());

            outFile << "Classical," << N << "," << classicalMean << "," << classicalStd << "\n";
            outFile << "GAN," << N << "," << ganMean << "," << ganStd << "\n";

            logger.info("Classical - Mean: " + std::to_string(classicalMean) + ", Std: " + std::to_string(classicalStd));
            logger.info("GAN - Mean: " + std::to_string(ganMean) + ", Std: " + std::to_string(ganStd));
        }

        outFile.close();
        logger.info("Results saved to strong_error_results.csv");
    }
};

int main(int argc, char* argv[]) {
    std::string configFile = "configs/default.yaml";
    if (argc > 1) {
        configFile = argv[1];
    }

    try {
        StrongErrorEvaluator evaluator(configFile);
        
        int numPaths = 1000;
        std::vector<int> timeSteps = {10, 20, 50, 100, 200};
        
        evaluator.evaluateClassicalVsGan(numPaths, timeSteps);
        
        std::cout << "Strong error evaluation completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}