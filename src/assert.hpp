#include <sstream>
#include <stdexcept>

namespace 
{
    inline void assert(const char* str, int line, const char* file, const char* func)
    {
        std::stringstream ss;
        ss << file << ":" << line << ": Assertion " << str << " in function " << func << " failed\n";
        throw std::logic_error(ss.str());
    }
}

#define DeepWorks_Assert(expr) \
{ if (!(expr)) assert(#expr, __LINE__, __FILE__, __func__); }
