#include <deepworks/placeholder.hpp>
#include <deepworks/call.hpp>

#include "placeholder_impl.hpp"

deepworks::Placeholder::Placeholder(const deepworks::Shape& shape,
                                    deepworks::Call         call,
                                    size_t                  port)
    : m_impl(new deepworks::Placeholder::Impl{shape, call, port}) {
}

deepworks::Placeholder::Placeholder(const deepworks::Shape& shape)
    : m_impl(new deepworks::Placeholder::Impl{shape, {}}) {
}

const deepworks::Shape& deepworks::Placeholder::shape() const {
    return m_impl->shape;
}

const deepworks::Placeholder::Impl& deepworks::Placeholder::impl() const {
    return *m_impl;
}

deepworks::Placeholder::Impl& deepworks::Placeholder::impl() {
    return *m_impl;
}
