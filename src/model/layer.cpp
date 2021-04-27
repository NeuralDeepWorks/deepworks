#include <deepworks/layer.hpp>

struct deepworks::Layer::Impl {
    deepworks::LayerInfo    m_info;
    deepworks::Placeholders m_inputs;
    deepworks::Placeholders m_outputs;
};

deepworks::Layer::Layer(deepworks::LayerInfo info,
                        deepworks::Placeholders inputs,
                        deepworks::Placeholders outputs)
    : m_impl(new deepworks::Layer::Impl{std::move(info),
                                        std::move(inputs),
                                        std::move(outputs)}) {
}

const std::string deepworks::Layer::name() const {
    return m_impl->m_info.name();
}

const std::string deepworks::Layer::type() const {
    return m_impl->m_info.type();
}

const deepworks::ParamMap& deepworks::Layer::params() const {
    return m_impl->m_info.params();
}

const deepworks::BufferMap& deepworks::Layer::buffers() const {
    return m_impl->m_info.buffers();
}

const deepworks::Placeholders& deepworks::Layer::inputs() const {
    return m_impl->m_inputs;
}

const deepworks::Placeholders& deepworks::Layer::outputs() const {
    return m_impl->m_outputs;
}

deepworks::Placeholders& deepworks::Layer::inputs() {
    return m_impl->m_inputs;
}

deepworks::Placeholders& deepworks::Layer::outputs() {
    return m_impl->m_outputs;
}
