#ifndef MCGAN_CORE_TYPES_HPP
#define MCGAN_CORE_TYPES_HPP

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <cmath>
#include <stdexcept>

namespace mcgan {
namespace core {

// ============================================================================
// Type Aliases
// ============================================================================

/// Floating point precision type
using Real = double;

/// Integer type for indices and counts
using Index = int;

/// Size type for containers
using Size = std::size_t;

/// Time type (in years)
using Time = Real;

/// Price/Value type
using Price = Real;

// ============================================================================
// Vector Types
// ============================================================================

/// Vector of real numbers
using RealVector = std::vector<Real>;

/// Vector of integers
using IntVector = std::vector<Index>;

/// 2D matrix as vector of vectors
using RealMatrix = std::vector<RealVector>;

/// Path type (time series of prices)
using Path = RealVector;

/// Collection of paths
using PathCollection = std::vector<Path>;

// ============================================================================
// Market Data Structures
// ============================================================================

/**
 * @brief Market parameters for Geometric Brownian Motion
 */
struct GBMParameters {
    Real S0;        ///< Initial stock price
    Real mu;        ///< Drift (expected return)
    Real sigma;     ///< Volatility
    Time T;         ///< Time to maturity
    Index N;        ///< Number of time steps
    
    /// Default constructor
    GBMParameters() 
        : S0(100.0), mu(0.05), sigma(0.2), T(1.0), N(100) {}
    
    /// Parameterized constructor
    GBMParameters(Real s0, Real drift, Real vol, Time maturity, Index steps)
        : S0(s0), mu(drift), sigma(vol), T(maturity), N(steps) {
        validate();
    }
    
    /// Validate parameters
    void validate() const {
        if (S0 <= 0.0) {
            throw std::invalid_argument("Initial price S0 must be positive");
        }
        if (sigma < 0.0) {
            throw std::invalid_argument("Volatility sigma must be non-negative");
        }
        if (T <= 0.0) {
            throw std::invalid_argument("Time to maturity T must be positive");
        }
        if (N <= 0) {
            throw std::invalid_argument("Number of steps N must be positive");
        }
    }
    
    /// Get time step size
    Real dt() const {
        return T / static_cast<Real>(N);
    }
};

/**
 * @brief Option contract specifications
 */
struct OptionSpec {
    enum class Type {
        CALL,
        PUT
    };
    
    enum class Style {
        EUROPEAN,
        AMERICAN,
        ASIAN,
        BARRIER,
        LOOKBACK
    };
    
    Type type;          ///< Call or Put
    Style style;        ///< Option style
    Price strike;       ///< Strike price
    Time maturity;      ///< Time to maturity
    Real barrier;       ///< Barrier level (for barrier options)
    
    /// Default constructor
    OptionSpec() 
        : type(Type::CALL), style(Style::EUROPEAN), 
          strike(100.0), maturity(1.0), barrier(0.0) {}
    
    /// Constructor for standard options
    OptionSpec(Type t, Style s, Price K, Time T)
        : type(t), style(s), strike(K), maturity(T), barrier(0.0) {
        validate();
    }
    
    /// Validate specification
    void validate() const {
        if (strike <= 0.0) {
            throw std::invalid_argument("Strike price must be positive");
        }
        if (maturity <= 0.0) {
            throw std::invalid_argument("Maturity must be positive");
        }
        if (style == Style::BARRIER && barrier <= 0.0) {
            throw std::invalid_argument("Barrier level must be positive for barrier options");
        }
    }
    
    /// Get option type as string
    std::string typeToString() const {
        return (type == Type::CALL) ? "Call" : "Put";
    }
    
    /// Get option style as string
    std::string styleToString() const {
        switch (style) {
            case Style::EUROPEAN: return "European";
            case Style::AMERICAN: return "American";
            case Style::ASIAN: return "Asian";
            case Style::BARRIER: return "Barrier";
            case Style::LOOKBACK: return "Lookback";
            default: return "Unknown";
        }
    }
};

// ============================================================================
// Statistical Result Types
// ============================================================================

/**
 * @brief Result with standard error (for Monte Carlo estimators)
 */
template<typename T>
struct ResultWithError {
    T value;            ///< Estimated value
    T std_error;        ///< Standard error of estimate
    Size num_samples;   ///< Number of samples used
    
    /// Default constructor
    ResultWithError() : value(T()), std_error(T()), num_samples(0) {}
    
    /// Parameterized constructor
    ResultWithError(T val, T err, Size n = 0) 
        : value(val), std_error(err), num_samples(n) {}
    
    /// Confidence interval at given level (default 95%)
    std::pair<T, T> confidenceInterval(Real confidence_level = 0.95) const {
        // Use normal approximation (z-score)
        Real z = 1.96;  // for 95% confidence
        if (confidence_level == 0.99) z = 2.576;
        else if (confidence_level == 0.90) z = 1.645;
        
        T margin = z * std_error;
        return {value - margin, value + margin};
    }
    
    /// Relative error (coefficient of variation)
    Real relativeError() const {
        if (std::abs(value) < 1e-10) return std::numeric_limits<Real>::infinity();
        return std::abs(std_error / value);
    }
};

/// Pricing result with standard error
using PricingResult = ResultWithError<Price>;

/**
 * @brief Statistics computed from a sample
 */
struct Statistics {
    Real mean;          ///< Sample mean
    Real variance;      ///< Sample variance
    Real std_dev;       ///< Standard deviation
    Real min_value;     ///< Minimum value
    Real max_value;     ///< Maximum value
    Size count;         ///< Sample size
    
    /// Default constructor
    Statistics() 
        : mean(0.0), variance(0.0), std_dev(0.0), 
          min_value(0.0), max_value(0.0), count(0) {}
    
    /// Compute statistics from data
    static Statistics compute(const RealVector& data) {
        Statistics stats;
        stats.count = data.size();
        
        if (stats.count == 0) return stats;
        
        // Mean
        Real sum = 0.0;
        stats.min_value = data[0];
        stats.max_value = data[0];
        
        for (const auto& x : data) {
            sum += x;
            stats.min_value = std::min(stats.min_value, x);
            stats.max_value = std::max(stats.max_value, x);
        }
        stats.mean = sum / static_cast<Real>(stats.count);
        
        // Variance
        Real sum_sq_diff = 0.0;
        for (const auto& x : data) {
            Real diff = x - stats.mean;
            sum_sq_diff += diff * diff;
        }
        stats.variance = sum_sq_diff / static_cast<Real>(stats.count);
        stats.std_dev = std::sqrt(stats.variance);
        
        return stats;
    }
    
    /// Standard error of the mean
    Real standardError() const {
        if (count == 0) return 0.0;
        return std_dev / std::sqrt(static_cast<Real>(count));
    }
};

// ============================================================================
// Payoff Functions
// ============================================================================

/// Payoff function type: takes path and returns payoff
using PayoffFunction = std::function<Price(const Path&)>;

/**
 * @brief Common payoff functions
 */
struct Payoffs {
    /// European call option payoff
    static Price europeanCall(Real spot, Real strike) {
        return std::max(spot - strike, 0.0);
    }
    
    /// European put option payoff
    static Price europeanPut(Real spot, Real strike) {
        return std::max(strike - spot, 0.0);
    }
    
    /// Asian call option payoff (arithmetic average)
    static Price asianCall(const Path& path, Real strike) {
        Real avg = 0.0;
        for (const auto& s : path) {
            avg += s;
        }
        avg /= static_cast<Real>(path.size());
        return std::max(avg - strike, 0.0);
    }
    
    /// Asian put option payoff (arithmetic average)
    static Price asianPut(const Path& path, Real strike) {
        Real avg = 0.0;
        for (const auto& s : path) {
            avg += s;
        }
        avg /= static_cast<Real>(path.size());
        return std::max(strike - avg, 0.0);
    }
    
    /// Lookback call option payoff
    static Price lookbackCall(const Path& path, Real strike) {
        Real max_price = *std::max_element(path.begin(), path.end());
        return std::max(max_price - strike, 0.0);
    }
    
    /// Lookback put option payoff
    static Price lookbackPut(const Path& path, Real strike) {
        Real min_price = *std::min_element(path.begin(), path.end());
        return std::max(strike - min_price, 0.0);
    }
    
    /// Up-and-out call barrier option
    static Price upAndOutCall(const Path& path, Real strike, Real barrier) {
        // Check if barrier was breached
        for (const auto& s : path) {
            if (s >= barrier) {
                return 0.0;  // Knocked out
            }
        }
        return europeanCall(path.back(), strike);
    }
};

// ============================================================================
// Training Configuration
// ============================================================================

/**
 * @brief Neural network training configuration
 */
struct TrainingConfig {
    Size batch_size;        ///< Batch size for training
    Size num_epochs;        ///< Number of training epochs
    Real learning_rate;     ///< Learning rate
    Real beta1;             ///< Adam optimizer beta1
    Real beta2;             ///< Adam optimizer beta2
    Size disc_steps;        ///< Discriminator steps per generator step
    Size latent_dim;        ///< Latent space dimension
    Size hidden_dim;        ///< Hidden layer dimension
    std::string device;     ///< Device (cpu/cuda)
    
    /// Default constructor with reasonable defaults
    TrainingConfig()
        : batch_size(64), num_epochs(5000), learning_rate(0.0002),
          beta1(0.5), beta2(0.999), disc_steps(1), 
          latent_dim(64), hidden_dim(256), device("cpu") {}
};

} // namespace core
} // namespace mcgan

#endif // MCGAN_CORE_TYPES_HPP