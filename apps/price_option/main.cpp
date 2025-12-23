#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "../include/mcgan/core/Config.hpp"
#include "../include/mcgan/core/Logger.hpp"
#include "../include/mcgan/mc/ClassicalSampler.hpp"
#include "../include/mcgan/mc/GanPathSampler.hpp"
#include "../include/mcgan/mc/Pricer.hpp"

// Option pricing application using Monte Carlo and GAN-based methods
class OptionPricingApp {
private:
    Config config;
    Logger logger;
    Pricer pricer;

public:
    OptionPricingApp(const std::string& configFile) 
        : config(configFile), logger("OptionPricer"), pricer() {}

    struct PricingResult {
        double price;
        double stdError;
        double computeTime;
        std::string method;
    };

    PricingResult priceEuropeanCall(double S0, double K, double r, double sigma, 
                                     double T, int numPaths, int timeSteps, 
                                     const std::string& method) {
        PricingResult result;
        result.method = method;
        
        auto start = std::chrono::high_resolution_clock::now();

        if (method == "classical") {
            auto [price, stdErr] = pricer.priceEuropeanCallClassical(
                S0, K, r, sigma, T, numPaths, timeSteps);
            result.price = price;
            result.stdError = stdErr;
        } else if (method == "gan") {
            auto [price, stdErr] = pricer.priceEuropeanCallGAN(
                S0, K, r, sigma, T, numPaths, timeSteps, "trained_gan_model.pt");
            result.price = price;
            result.stdError = stdErr;
        } else if (method == "blackscholes") {
            result.price = pricer.blackScholesCall(S0, K, r, sigma, T);
            result.stdError = 0.0;
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.computeTime = std::chrono::duration<double>(end - start).count();

        return result;
    }

    PricingResult priceAsianCall(double S0, double K, double r, double sigma, 
                                  double T, int numPaths, int timeSteps, 
                                  const std::string& method) {
        PricingResult result;
        result.method = method;
        
        auto start = std::chrono::high_resolution_clock::now();

        if (method == "classical") {
            auto [price, stdErr] = pricer.priceAsianCallClassical(
                S0, K, r, sigma, T, numPaths, timeSteps);
            result.price = price;
            result.stdError = stdErr;
        } else if (method == "gan") {
            auto [price, stdErr] = pricer.priceAsianCallGAN(
                S0, K, r, sigma, T, numPaths, timeSteps, "trained_gan_model.pt");
            result.price = price;
            result.stdError = stdErr;
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.computeTime = std::chrono::duration<double>(end - start).count();

        return result;
    }

    void runComparison() {
        double S0 = config.getDouble("S0", 100.0);
        double K = config.getDouble("K", 100.0);
        double r = config.getDouble("r", 0.05);
        double sigma = config.getDouble("sigma", 0.2);
        double T = config.getDouble("T", 1.0);
        int numPaths = config.getInt("num_paths", 100000);
        int timeSteps = config.getInt("time_steps", 100);

        logger.info("Starting option pricing comparison");
        logger.info("Parameters: S0=" + std::to_string(S0) + ", K=" + std::to_string(K) +
                    ", r=" + std::to_string(r) + ", sigma=" + std::to_string(sigma) +
                    ", T=" + std::to_string(T));

        // European Call Option
        std::cout << "\n=== European Call Option ===" << std::endl;
        
        auto bsResult = priceEuropeanCall(S0, K, r, sigma, T, numPaths, timeSteps, "blackscholes");
        std::cout << "Black-Scholes Price: " << bsResult.price << std::endl;

        auto classicalResult = priceEuropeanCall(S0, K, r, sigma, T, numPaths, timeSteps, "classical");
        std::cout << "Classical MC Price: " << classicalResult.price 
                  << " ± " << classicalResult.stdError 
                  << " (Time: " << classicalResult.computeTime << "s)" << std::endl;
        std::cout << "Error vs BS: " << std::abs(classicalResult.price - bsResult.price) << std::endl;

        auto ganResult = priceEuropeanCall(S0, K, r, sigma, T, numPaths, timeSteps, "gan");
        std::cout << "GAN MC Price: " << ganResult.price 
                  << " ± " << ganResult.stdError 
                  << " (Time: " << ganResult.computeTime << "s)" << std::endl;
        std::cout << "Error vs BS: " << std::abs(ganResult.price - bsResult.price) << std::endl;
        std::cout << "Speedup: " << classicalResult.computeTime / ganResult.computeTime << "x" << std::endl;

        // Asian Call Option
        std::cout << "\n=== Asian Call Option ===" << std::endl;
        
        auto classicalAsian = priceAsianCall(S0, K, r, sigma, T, numPaths, timeSteps, "classical");
        std::cout << "Classical MC Price: " << classicalAsian.price 
                  << " ± " << classicalAsian.stdError 
                  << " (Time: " << classicalAsian.computeTime << "s)" << std::endl;

        auto ganAsian = priceAsianCall(S0, K, r, sigma, T, numPaths, timeSteps, "gan");
        std::cout << "GAN MC Price: " << ganAsian.price 
                  << " ± " << ganAsian.stdError 
                  << " (Time: " << ganAsian.computeTime << "s)" << std::endl;
        std::cout << "Speedup: " << classicalAsian.computeTime / ganAsian.computeTime << "x" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::string configFile = "configs/default.yaml";
    if (argc > 1) {
        configFile = argv[1];
    }

    try {
        OptionPricingApp app(configFile);
        app.runComparison();
        
        std::cout << "\nOption pricing completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}