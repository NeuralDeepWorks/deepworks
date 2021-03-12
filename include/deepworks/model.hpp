#pragma once

#include <vector>
#include <memory>

#include <deepworks/placeholder.hpp>
#include <deepworks/layer.hpp>
#include <deepworks/tensor.hpp>

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

    void compile(int batch_size);

    std::vector<Tensor> forward(const std::vector<Tensor> & inputs);
    std::vector<Tensor> backward(const std::vector<Tensor>& inputs);

private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace deepworks
