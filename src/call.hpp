#pragma once

#include <memory> // shared_ptr

#include "layer_info.hpp"

namespace deepworks {

class Placeholder;
struct Call {
    Call() = default;
    explicit Call(LayerInfo&&);

    void pass(std::vector<Placeholder>&& args);

    Placeholder create(const Shape& shape);

    struct Impl;
    const Impl& impl() const;
    std::shared_ptr<Impl> m_impl;
};

} // namespace deepworks
