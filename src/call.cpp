#include <deepworks/placeholder.hpp>

#include "call.hpp"
#include "call_priv.hpp"

deepworks::Call::Call(deepworks::LayerInfo&& info)
    : m_priv(new Call::Priv{std::move(info)}) {
}

void deepworks::Call::pass(std::vector<deepworks::Placeholder>&& args) {
    m_priv->args = std::move(args);
}

deepworks::Placeholder deepworks::Call::create(const deepworks::Shape& shape) {
    return deepworks::Placeholder{shape, *this};
};

const deepworks::Call::Priv& deepworks::Call::priv() const {
    return *m_priv;
}
