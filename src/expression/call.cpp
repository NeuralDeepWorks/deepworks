#include <deepworks/placeholder.hpp>
#include <deepworks/layer_info.hpp>
#include <deepworks/call.hpp>

#include "call_impl.hpp"

deepworks::Call::Call(const deepworks::LayerInfo& info)
    : m_impl(new Call::Impl{info}) {
}

void deepworks::Call::pass(deepworks::Placeholders&& args) {
    m_impl->args = std::move(args);
}

deepworks::Placeholder deepworks::Call::create(const deepworks::Shape& shape) {
    return deepworks::Placeholder{shape, *this};
};

const deepworks::Call::Impl& deepworks::Call::impl() const {
    return *m_impl;
}

deepworks::Call::Impl& deepworks::Call::impl() {
    return *m_impl;
}
