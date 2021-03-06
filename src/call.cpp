#include <deepworks/placeholder.hpp>

#include "call.hpp"
#include "call_impl.hpp"

deepworks::Call::Call(deepworks::LayerInfo&& info)
    : m_impl(new Call::Impl{std::move(info)}) {
}

void deepworks::Call::pass(std::vector<deepworks::Placeholder>&& args) {
    m_impl->args = std::move(args);
}

deepworks::Placeholder deepworks::Call::create(const deepworks::Shape& shape) {
    return deepworks::Placeholder{shape, *this};
};

const deepworks::Call::Impl& deepworks::Call::impl() const {
    return *m_impl;
}
