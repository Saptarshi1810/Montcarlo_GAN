#ifndef MCGAN_CORE_LOGGER_HPP
#define MCGAN_CORE_LOGGER_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <memory>

namespace mcgan {
namespace core {

/**
 * @brief Log severity levels
 */
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

/**
 * @brief Thread-safe logger for console and file output
 * 
 * Provides formatted logging with timestamps, severity levels, and context.
 * Supports both console and file output with configurable verbosity.
 */
class Logger {
private:
    std::string name_;
    LogLevel level_;
    std::ofstream file_stream_;
    bool console_output_;
    bool file_output_;
    std::mutex mutex_;
    
    static std::shared_ptr<Logger> global_logger_;
    static std::mutex global_mutex_;

public:
    /**
     * @brief Constructor with logger name
     * @param name Logger identifier (appears in log messages)
     * @param level Minimum log level to output
     */
    explicit Logger(const std::string& name = "MCGAN", 
                   LogLevel level = LogLevel::INFO)
        : name_(name), level_(level), console_output_(true), file_output_(false) {}

    /**
     * @brief Destructor - ensures file is closed
     */
    ~Logger() {
        if (file_stream_.is_open()) {
            file_stream_.close();
        }
    }

    /**
     * @brief Enable file logging
     * @param filename Path to log file
     * @param append If true, append to existing file; otherwise overwrite
     */
    void enableFileOutput(const std::string& filename, bool append = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (file_stream_.is_open()) {
            file_stream_.close();
        }
        
        auto mode = append ? std::ios::app : std::ios::out;
        file_stream_.open(filename, mode);
        
        if (!file_stream_.is_open()) {
            std::cerr << "Failed to open log file: " << filename << std::endl;
            file_output_ = false;
        } else {
            file_output_ = true;
        }
    }

    /**
     * @brief Disable file logging
     */
    void disableFileOutput() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_stream_.is_open()) {
            file_stream_.close();
        }
        file_output_ = false;
    }

    /**
     * @brief Enable/disable console output
     * @param enable true to enable console output
     */
    void setConsoleOutput(bool enable) {
        console_output_ = enable;
    }

    /**
     * @brief Set minimum log level
     * @param level New minimum log level
     */
    void setLevel(LogLevel level) {
        level_ = level;
    }

    /**
     * @brief Get current log level
     * @return Current minimum log level
     */
    LogLevel getLevel() const {
        return level_;
    }

    /**
     * @brief Log debug message
     * @param message Message to log
     */
    void debug(const std::string& message) {
        log(LogLevel::DEBUG, message);
    }

    /**
     * @brief Log info message
     * @param message Message to log
     */
    void info(const std::string& message) {
        log(LogLevel::INFO, message);
    }

    /**
     * @brief Log warning message
     * @param message Message to log
     */
    void warning(const std::string& message) {
        log(LogLevel::WARNING, message);
    }

    /**
     * @brief Log error message
     * @param message Message to log
     */
    void error(const std::string& message) {
        log(LogLevel::ERROR, message);
    }

    /**
     * @brief Log critical message
     * @param message Message to log
     */
    void critical(const std::string& message) {
        log(LogLevel::CRITICAL, message);
    }

    /**
     * @brief Log message with specified level
     * @param level Log level for this message
     * @param message Message to log
     */
    void log(LogLevel level, const std::string& message) {
        if (level < level_) {
            return;
        }

        std::string formatted = formatMessage(level, message);
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (console_output_) {
            if (level >= LogLevel::ERROR) {
                std::cerr << formatted << std::endl;
            } else {
                std::cout << formatted << std::endl;
            }
        }
        
        if (file_output_ && file_stream_.is_open()) {
            file_stream_ << formatted << std::endl;
            file_stream_.flush();
        }
    }

    /**
     * @brief Get global logger instance
     * @return Shared pointer to global logger
     */
    static std::shared_ptr<Logger> getGlobalLogger() {
        std::lock_guard<std::mutex> lock(global_mutex_);
        if (!global_logger_) {
            global_logger_ = std::make_shared<Logger>("GLOBAL");
        }
        return global_logger_;
    }

    /**
     * @brief Set global logger instance
     * @param logger New global logger
     */
    static void setGlobalLogger(std::shared_ptr<Logger> logger) {
        std::lock_guard<std::mutex> lock(global_mutex_);
        global_logger_ = logger;
    }

private:
    /**
     * @brief Get current timestamp as string
     * @return Formatted timestamp
     */
    std::string getTimestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        
        return ss.str();
    }

    /**
     * @brief Convert log level to string
     * @param level Log level
     * @return String representation
     */
    std::string levelToString(LogLevel level) const {
        switch (level) {
            case LogLevel::DEBUG:    return "DEBUG";
            case LogLevel::INFO:     return "INFO";
            case LogLevel::WARNING:  return "WARNING";
            case LogLevel::ERROR:    return "ERROR";
            case LogLevel::CRITICAL: return "CRITICAL";
            default:                 return "UNKNOWN";
        }
    }

    /**
     * @brief Get ANSI color code for log level
     * @param level Log level
     * @return ANSI color code
     */
    std::string getColorCode(LogLevel level) const {
        if (!console_output_) {
            return "";
        }
        
        switch (level) {
            case LogLevel::DEBUG:    return "\033[36m";  // Cyan
            case LogLevel::INFO:     return "\033[32m";  // Green
            case LogLevel::WARNING:  return "\033[33m";  // Yellow
            case LogLevel::ERROR:    return "\033[31m";  // Red
            case LogLevel::CRITICAL: return "\033[35m";  // Magenta
            default:                 return "\033[0m";   // Reset
        }
    }

    /**
     * @brief Format log message with timestamp and context
     * @param level Log level
     * @param message Message to format
     * @return Formatted message
     */
    std::string formatMessage(LogLevel level, const std::string& message) const {
        std::stringstream ss;
        
        std::string color = getColorCode(level);
        std::string reset = console_output_ ? "\033[0m" : "";
        
        ss << "[" << getTimestamp() << "] "
           << color << "[" << levelToString(level) << "]" << reset
           << " [" << name_ << "] "
           << message;
        
        return ss.str();
    }
};

// Static member initialization
inline std::shared_ptr<Logger> Logger::global_logger_ = nullptr;
inline std::mutex Logger::global_mutex_;

// Convenience macros for global logger
#define LOG_DEBUG(msg) mcgan::core::Logger::getGlobalLogger()->debug(msg)
#define LOG_INFO(msg) mcgan::core::Logger::getGlobalLogger()->info(msg)
#define LOG_WARNING(msg) mcgan::core::Logger::getGlobalLogger()->warning(msg)
#define LOG_ERROR(msg) mcgan::core::Logger::getGlobalLogger()->error(msg)
#define LOG_CRITICAL(msg) mcgan::core::Logger::getGlobalLogger()->critical(msg)

} // namespace core
} // namespace mcgan

#endif // MCGAN_CORE_LOGGER_HPP