#pragma once

#include <random>

namespace deepworks {
namespace detail {

inline std::mt19937& generator() {
    static std::mt19937 gen{std::random_device{}()};
    return gen;
}

} // namespace detail
} // namespace deepworks
