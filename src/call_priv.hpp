#include "call.hpp"

namespace deepworks {

struct Call::Priv {
    deepworks::LayerInfo     info;
    std::vector<Placeholder> args;
};

} // namespace deepworks
