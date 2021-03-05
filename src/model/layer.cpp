#include <deepworks/layer.hpp>

struct deepworks::Layer::Priv {
    deepworks::LayerInfo    m_info;
    deepworks::Placeholders m_inputs;
    deepworks::Placeholders m_outputs;
};

deepworks::Layer::Layer(deepworks::LayerInfo info,
                        deepworks::Placeholders inputs,
                        deepworks::Placeholders outputs)
    : m_priv(new deepworks::Layer::Priv{std::move(info),
                                        std::move(inputs),
                                        std::move(outputs)}) {
}

const std::string deepworks::Layer::name() const {
    return m_priv->m_info.name();
}

const std::string deepworks::Layer::type() const {
    return m_priv->m_info.type();
}

const deepworks::Placeholders& deepworks::Layer::inputs() const {
    return m_priv->m_inputs;
}

const deepworks::Placeholders& deepworks::Layer::outputs() const {
    return m_priv->m_outputs;
}

deepworks::Placeholders& deepworks::Layer::inputs() {
    return m_priv->m_inputs;
}

deepworks::Placeholders& deepworks::Layer::outputs() {
    return m_priv->m_outputs;
}
