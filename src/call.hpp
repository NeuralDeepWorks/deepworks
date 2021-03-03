#pragma once

#include <memory> // shared_ptr

#include "layer_info.hpp"

namespace deepworks {

classPlaceholder;
struct Call {
    Call() = default;
    explicit Call(LayerInfo&&);

    void pass(std::vector<Placeholder>&& args);

    Placeholder create(const Shape& shape);

    struct Priv;
    const Priv& priv() const;
    std::shared_ptr<Priv> m_priv;
};

} // namespace deepworks
