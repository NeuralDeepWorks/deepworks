#pragma once

#include <vector>
#include <memory>

#include <deepworks/placeholder.hpp>
#include <deepworks/layer.hpp>

namespace deepworks {

class Model {
public:
    Model(Placeholder  in,  Placeholder  out );
    Model(Placeholders ins, Placeholders outs);

    const Placeholders& inputs()  const;
    const Placeholders& outputs() const;
    const Layers      & layers()  const;
          Layers      & layers();

    Layer getLayer(const std::string& name);

private:
    class Priv;
    std::shared_ptr<Priv> m_priv;
};

} // namespace deepworks
