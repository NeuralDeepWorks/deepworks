#pragma once

#include <deepworks/tensor.hpp>

#include "runtime/cpu/layers/cpulayer.hpp"

#include "util/assert.hpp"

namespace deepworks {
namespace cpu {

class CPULinear : public ICPULayer {
public:
    CPULinear(LayerInfo&& info);
    virtual void forward(const std::vector<Tensor>& inputs,
                               std::vector<Tensor>& outputs) override;

    virtual void updateGradients(const std::vector<Tensor>& inputs,
                                 const std::vector<Tensor>& grad_outputs) override;

    virtual void backward(const std::vector<Tensor>& inputs,
                          const std::vector<Tensor>& outputs,
                          const std::vector<Tensor>& grad_outputs,
                                std::vector<Tensor>& grad_inputs) override;
private:
    void validate(const std::vector<Tensor>& inputs,
                  const std::vector<Tensor>& outputs);

    deepworks::Tensor m_W, m_b;
    deepworks::Tensor m_gradW, m_gradb;
};

} // namespace cpu
} // namespace deepworks
