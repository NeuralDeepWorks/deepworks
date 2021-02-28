#include <deepworks/placeholder.hpp>

#include "call.hpp"
#include "placeholder_priv.hpp"

deepworks::Placeholder::Placeholder(const deepworks::Shape& shape,
                                    deepworks::Call call)
    : m_priv(new deepworks::Placeholder::Priv{shape, call}) {
    }

deepworks::Placeholder::Placeholder(const deepworks::Shape& shape)
    : m_priv(new deepworks::Placeholder::Priv{shape, {}}) {
    }

const deepworks::Shape& deepworks::Placeholder::shape() const {
    return m_priv->shape;
}

const deepworks::Placeholder::Priv& deepworks::Placeholder::priv() const {
    return *m_priv;
}
