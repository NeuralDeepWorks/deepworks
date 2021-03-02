#include "call.hpp"

namespace deepworks {

struct Call::Priv {
    deepworks::LayerInfo info;
    Placeholders         args;
};

} // namespace deepworks
