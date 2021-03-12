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
    ICPULayer(LayerInfo info);

    virtual void forward(const std::vector<Tensor>& inputs,
                               std::vector<Tensor>& outputs) = 0;

private:
    LayerInfo m_info;
};

} // namespace cpu
} // namespace deepworks
