#pragma once

#include <vector>

#include <deepworks/placeholder.hpp>

#include "layer_info.hpp"

namespace deepworks {

// NB: User have access to this structure
// to read name, type and freeze/unfreeze parameters.
struct Layer {
    const std::string name()   const { return info.name();   }
    const std::string type()   const { return info.type();   }
 /* const Parameters  params() const { return info.params(); } */
    
    LayerInfo info;

    Placeholders inputs;
    Placeholders outputs;
};

using Layers = std::vector<Layer>;

class Model {
public:
    Model(Placeholder  in,  Placeholder  out );
    Model(Placeholders ins, Placeholders outs);

    const Placeholders& inputs()  const;
    const Placeholders& outputs() const;
    const Layers      & layers()  const;

private:
    void unroll(const Placeholders& ins, const Placeholders& outs);

    Placeholders m_inputs;
    Placeholders m_outputs;

    std::vector<Layer> m_layers;
};

} // namespace deepworks
