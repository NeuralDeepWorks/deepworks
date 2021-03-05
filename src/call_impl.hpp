#include "call.hpp"

namespace deepworks {

struct Call::Impl {
    deepworks::LayerInfo     info;
    std::vector<Placeholder> args;
};

} // namespace deepworks
