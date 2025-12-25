#ifndef MCGAN_CORE_CONFIG_HPP
#define MCGAN_CORE_CONFIG_HPP

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace mcgan {
namespace core {

/**
 * @brief Configuration manager for loading and accessing configuration parameters
 * 
 * Supports YAML configuration files with hierarchical structure.
 * Provides type-safe access to configuration values with default fallbacks.
 */
class Config {
private:
    YAML::Node config_;
    std::string config_file_;
    std::map<std::string, std::string> overrides_;

public:
    /**
     * @brief Default constructor - creates empty config
     */
    Config() = default;

    /**
     * @brief Constructor that loads configuration from file
     * @param config_file Path to YAML configuration file
     */
    explicit Config(const std::string& config_file) : config_file_(config_file) {
        load(config_file);
    }

    /**
     * @brief Load configuration from YAML file
     * @param config_file Path to configuration file
     * @throws std::runtime_error if file cannot be loaded
     */
    void load(const std::string& config_file) {
        try {
            config_ = YAML::LoadFile(config_file);
            config_file_ = config_file;
        } catch (const YAML::Exception& e) {
            throw std::runtime_error("Failed to load config file '" + config_file + "': " + e.what());
        }
    }

    /**
     * @brief Get string value from configuration
     * @param key Configuration key (supports dot notation for nested keys)
     * @param default_value Default value if key not found
     * @return Configuration value or default
     */
    std::string getString(const std::string& key, const std::string& default_value = "") const {
        // Check overrides first
        auto it = overrides_.find(key);
        if (it != overrides_.end()) {
            return it->second;
        }

        try {
            YAML::Node node = getNode(key);
            if (node && !node.IsNull()) {
                return node.as<std::string>();
            }
        } catch (const YAML::Exception&) {
            // Fall through to return default
        }
        return default_value;
    }

    /**
     * @brief Get integer value from configuration
     * @param key Configuration key
     * @param default_value Default value if key not found
     * @return Configuration value or default
     */
    int getInt(const std::string& key, int default_value = 0) const {
        auto it = overrides_.find(key);
        if (it != overrides_.end()) {
            return std::stoi(it->second);
        }

        try {
            YAML::Node node = getNode(key);
            if (node && !node.IsNull()) {
                return node.as<int>();
            }
        } catch (const YAML::Exception&) {
            // Fall through to return default
        }
        return default_value;
    }

    /**
     * @brief Get double value from configuration
     * @param key Configuration key
     * @param default_value Default value if key not found
     * @return Configuration value or default
     */
    double getDouble(const std::string& key, double default_value = 0.0) const {
        auto it = overrides_.find(key);
        if (it != overrides_.end()) {
            return std::stod(it->second);
        }

        try {
            YAML::Node node = getNode(key);
            if (node && !node.IsNull()) {
                return node.as<double>();
            }
        } catch (const YAML::Exception&) {
            // Fall through to return default
        }
        return default_value;
    }

    /**
     * @brief Get boolean value from configuration
     * @param key Configuration key
     * @param default_value Default value if key not found
     * @return Configuration value or default
     */
    bool getBool(const std::string& key, bool default_value = false) const {
        auto it = overrides_.find(key);
        if (it != overrides_.end()) {
            return it->second == "true" || it->second == "1";
        }

        try {
            YAML::Node node = getNode(key);
            if (node && !node.IsNull()) {
                return node.as<bool>();
            }
        } catch (const YAML::Exception&) {
            // Fall through to return default
        }
        return default_value;
    }

    /**
     * @brief Get vector of integers from configuration
     * @param key Configuration key
     * @param default_value Default value if key not found
     * @return Configuration value or default
     */
    std::vector<int> getIntVector(const std::string& key, 
                                   const std::vector<int>& default_value = {}) const {
        try {
            YAML::Node node = getNode(key);
            if (node && node.IsSequence()) {
                return node.as<std::vector<int>>();
            }
        } catch (const YAML::Exception&) {
            // Fall through to return default
        }
        return default_value;
    }

    /**
     * @brief Get vector of doubles from configuration
     * @param key Configuration key
     * @param default_value Default value if key not found
     * @return Configuration value or default
     */
    std::vector<double> getDoubleVector(const std::string& key, 
                                        const std::vector<double>& default_value = {}) const {
        try {
            YAML::Node node = getNode(key);
            if (node && node.IsSequence()) {
                return node.as<std::vector<double>>();
            }
        } catch (const YAML::Exception&) {
            // Fall through to return default
        }
        return default_value;
    }

    /**
     * @brief Set override value (takes precedence over config file)
     * @param key Configuration key
     * @param value Override value
     */
    void setOverride(const std::string& key, const std::string& value) {
        overrides_[key] = value;
    }

    /**
     * @brief Check if key exists in configuration
     * @param key Configuration key
     * @return true if key exists
     */
    bool hasKey(const std::string& key) const {
        if (overrides_.find(key) != overrides_.end()) {
            return true;
        }
        
        try {
            YAML::Node node = getNode(key);
            return node && !node.IsNull();
        } catch (const YAML::Exception&) {
            return false;
        }
    }

    /**
     * @brief Get configuration file path
     * @return Path to loaded configuration file
     */
    std::string getConfigFile() const {
        return config_file_;
    }

    /**
     * @brief Print all configuration values
     */
    void print() const {
        std::cout << "Configuration from: " << config_file_ << std::endl;
        std::cout << config_ << std::endl;
        
        if (!overrides_.empty()) {
            std::cout << "\nOverrides:" << std::endl;
            for (const auto& [key, value] : overrides_) {
                std::cout << "  " << key << ": " << value << std::endl;
            }
        }
    }

private:
    /**
     * @brief Get YAML node for given key (supports dot notation)
     * @param key Configuration key (e.g., "model.layers.0")
     * @return YAML node
     */
    YAML::Node getNode(const std::string& key) const {
        YAML::Node node = config_;
        std::istringstream ss(key);
        std::string token;
        
        while (std::getline(ss, token, '.')) {
            if (!node) {
                return YAML::Node();
            }
            node = node[token];
        }
        
        return node;
    }
};

} // namespace core
} // namespace mcgan

#endif // MCGAN_CORE_CONFIG_HPP