#pragma once

#include <deepworks/tensor.hpp>

#include "runtime/cpu/layers/cpulayer.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

#include "util/assert.hpp"

namespace deepworks {
namespace cpu {

class CPUSoftmax : public ICPULayer {
public:
    using ICPULayer::ICPULayer;

    virtual void forward(const std::vector<Tensor>& inputs,
                               std::vector<Tensor>& outputs) override;

    virtual void backward(const std::vector<Tensor>& inputs,
                          const std::vector<Tensor>& outputs,
                          const std::vector<Tensor>& grad_outputs,
                                std::vector<Tensor>& grad_inputs) override;
private:
    void validate(const std::vector<Tensor>& inputs,
                  const std::vector<Tensor>& outputs);
};

} // namespace cpu
} // namespace deepworks
