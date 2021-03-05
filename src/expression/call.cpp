#include <deepworks/placeholder.hpp>

#include "call.hpp"
#include "call_priv.hpp"

deepworks::Call::Call(const deepworks::LayerInfo& info)
    // FIXME: Copy LayerInfo can be underpeformance
    : m_priv(new Call::Priv{info}) {
}

void deepworks::Call::pass(Placeholders&& args) {
    m_priv->args = std::move(args);
}

deepworks::Placeholder deepworks::Call::create(const deepworks::Shape& shape) {
    return deepworks::Placeholder{shape, *this};
};

const deepworks::Call::Priv& deepworks::Call::priv() const {
    return *m_priv;
}

deepworks::Call::Priv& deepworks::Call::priv() {
    return *m_priv;
}
