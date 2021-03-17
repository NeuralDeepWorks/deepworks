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

    // Execution API
    void compile();
    void forward (const Tensor& input, Tensor& outputs);
    void backward(const Tensor& input, Tensor& outputs);
    void forward (const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);
    void backward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);

private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace deepworks
