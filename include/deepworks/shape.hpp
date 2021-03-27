#pragma once

#include <vector>
#include <ostream>

namespace deepworks {

using Shape = std::vector<int>;

} // namespace deepworks;

// NB: Outside of namespace to make it available for user.
inline std::ostream& operator<<(std::ostream& os, const deepworks::Shape& shape) {
    if (shape.empty()) {
        os << "{ }";
        return os;
    }

    os << "{";
    for (int i = 0; i < shape.size() - 1; ++i) {
        os << shape[i] << ", ";
    }
    os << shape.back();
    os << "}";
    return os;
}
