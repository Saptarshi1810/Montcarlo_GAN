#include "Config.hpp"

namespace mcgan {
namespace core {

// Config class is fully implemented in the header file.
// This is intentional as it's a template-heavy class with YAML dependencies.
// 
// If you need to add any additional non-inline methods, they can be added here.
// For now, this file serves as a placeholder to maintain project structure.

// Example of how to add a non-inline method if needed:
/*
void Config::saveToFile(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    out << config_;
    out.close();
}
*/

} // namespace core
} // namespace mcgan