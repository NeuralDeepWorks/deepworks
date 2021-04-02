#pragma once

#include <memory>

#include <deepworks/layer_info.hpp>
#include <deepworks/tensor.hpp>

namespace deepworks {
namespace cpu {

class ICPULayer {
public:
    using Ptr = std::shared_ptr<ICPULayer>;
    static ICPULayer::Ptr create(deepworks::LayerInfo info);
    ICPULayer(LayerInfo&& info);

    virtual void forward(const std::vector<Tensor>& inputs,
                               std::vector<Tensor>& outputs) = 0;

    // NB: Update gradients by layer parameters. (by default do nothing)
    virtual void updateGradients(const std::vector<Tensor>& inputs,
                                 const std::vector<Tensor>& grad_outputs);

    virtual void backward(const std::vector<Tensor>& inputs,
                          const std::vector<Tensor>& outputs,
                          const std::vector<Tensor>& grad_outputs,
                                std::vector<Tensor>& grad_inputs) = 0;

protected:
    LayerInfo m_info;
};

} // namespace cpu
} // namespace deepworks
