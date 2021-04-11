#pragma once

#include <vector>
#include <memory>

#include <deepworks/placeholder.hpp>
#include <deepworks/layer.hpp>
#include <deepworks/tensor.hpp>
#include <deepworks/parameter.hpp>

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
    Parameters& params();

    void train(bool mode);

    // Execution API
    void compile();
    void forward(const Tensor& input, Tensor& outputs);
    void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);

    void backward(const Tensor& input,
                  const Tensor& output,
                  const Tensor& grad_output);

    void backward(const std::vector<Tensor>& inputs,
                  const std::vector<Tensor>& outputs,
                  const std::vector<Tensor>& grad_outputs);

private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace deepworks
