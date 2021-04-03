#include "util/assert.hpp"

deepworks::details::DeepWorksException::DeepWorksException(int line,
                                                           const std::string& filename,
                                                           const std::string& func) noexcept
    : m_impl(new deepworks::details::DeepWorksException::Impl()) {
    m_impl->line     = line;
    m_impl->filename = filename;
    m_impl->func     = func;
    m_impl->stream << "Failed: " << m_impl->filename << " in function: "
                                 << m_impl->func     << " in line: "
                                 << m_impl->line     << std::endl;
}

const char* deepworks::details::DeepWorksException::what() const noexcept {
    m_impl->message = m_impl->stream.str();
    return m_impl->message.c_str();
}
