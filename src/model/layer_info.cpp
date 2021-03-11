#include <deepworks/layer_info.hpp>

deepworks::LayerInfo::LayerInfo(std::string name, std::string type)
    : m_impl(new deepworks::LayerInfo::Impl{name, type}) {
}

const deepworks::LayerInfo::Impl& deepworks::LayerInfo::impl() const {
    return *m_impl;
}

deepworks::LayerInfo::Impl& deepworks::LayerInfo::impl() {
    return *m_impl;
}
