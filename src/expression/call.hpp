#pragma once

#include <memory> // shared_ptr

#include <deepworks/layer_info.hpp>

namespace deepworks {

class Placeholder;
struct Call {
    Call() = default;
    explicit Call(const LayerInfo&);

    void pass(std::vector<Placeholder>&& args);

    Placeholder create(const Shape& shape);

    struct Impl;
    const Impl& impl() const;
          Impl& impl();
    std::shared_ptr<Impl> m_priv;
};

using Calls = std::vector<Call>;

} // namespace deepworks
