#include "call.hpp"

namespace deepworks {

struct Call::Impl {
    LayerInfo    info;
    Placeholders args;
};

} // namespace deepworks
