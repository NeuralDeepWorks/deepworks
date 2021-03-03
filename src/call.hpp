#pragma once

#include <memory> // shared_ptr

#include <deepworks/layer_info.hpp>

namespace deepworks {

struct Placeholder;
struct Call {
    Call() = default;
    explicit Call(const LayerInfo&);

    void pass(std::vector<Placeholder>&& args);

    Placeholder create(const Shape& shape);

    struct Priv;
    const Priv& priv() const;
          Priv& priv();
    std::shared_ptr<Priv> m_priv;
};

using Calls = std::vector<Call>;

} // namespace deepworks
