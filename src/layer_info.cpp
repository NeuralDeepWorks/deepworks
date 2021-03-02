#include <deepworks/layer_info.hpp>

deepworks::LayerInfo::LayerInfo(std::string name, std::string type)
    : m_priv(new deepworks::LayerInfo::Priv{name, type}) {
    }

const deepworks::LayerInfo::Priv& deepworks::LayerInfo::priv() const {
    return *m_priv;
}

deepworks::LayerInfo::Priv& deepworks::LayerInfo::priv() {
    return *m_priv;
}
