#pragma once

#include <optional>

#include <deepworks/call.hpp>

namespace deepworks {

struct Placeholder::Impl {
    Shape shape;
    // NB: The creator, empty optional if it's input.
    std::optional<Call> call;
    // NB: The output port from producer. (invalid value by default)
    size_t port = std::numeric_limits<size_t>::max();
};

} // namespace deepworks
