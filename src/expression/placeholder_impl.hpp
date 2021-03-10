#pragma once

#include <optional>

#include "call.hpp"

namespace deepworks {

struct Placeholder::Impl {
    Shape shape;
    // NB: The creator, empty optional if it's input.
    std::optional<Call> call;
};

} // namespace deepworks
