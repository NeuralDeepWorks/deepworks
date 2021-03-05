#pragma once

#include <string>
#include <memory>

#include <deepworks/placeholder.hpp>
#include <deepworks/layer_info.hpp>

namespace deepworks {

class Layer {
public:
    // FIXME: LayerInfo should be removed from public header
    // and this constructor as well !!!
    Layer(LayerInfo info, Placeholders inputs, Placeholders outputs);

    const std::string   name()    const;
    const std::string   type()    const;
    const Placeholders& inputs()  const;
    const Placeholders& outputs() const;

    Placeholders& inputs();
    Placeholders& outputs();
    
private:
    struct Priv;
    std::shared_ptr<Priv> m_priv;
};

using Layers = std::vector<Layer>;

} // namespace deepworks
