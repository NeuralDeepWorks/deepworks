#pragma once

#include <sstream>
#include <stdexcept>
#include <memory>

namespace deepworks {
namespace utils {

class DeepWorksException : public std::exception {
public:
    DeepWorksException(int line, const std::string& filename, const std::string& func) noexcept;
    DeepWorksException(const DeepWorksException& that) noexcept = default;

    const char* what() const noexcept override;

    template <typename T>
    DeepWorksException& operator<<(const T& arg) noexcept {
        m_impl->stream << arg;
        return *this;
    }

    ~DeepWorksException() noexcept override { };

private:
    struct Impl {
        int               line;
        std::string       filename;
        std::string       func;
        std::stringstream stream;
        std::string       message;
    };

    std::shared_ptr<Impl> m_impl;
};

} // namespace utils
} // namespace deepworks

#define DeepWorks_Throw() \
    throw deepworks::utils::DeepWorksException(__LINE__, __FILE__, __func__)

#define DeepWorks_Assert(expr) \
{ if (!(expr)) DeepWorks_Throw() << "Assertion failed: " << #expr; }
