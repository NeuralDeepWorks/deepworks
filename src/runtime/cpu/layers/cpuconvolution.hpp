#pragma once

#include <deepworks/tensor.hpp>
#include "runtime/cpu/layers/cpulayer.hpp"
#include "util/assert.hpp"

namespace deepworks {
namespace cpu {

class CPUConvolution : public ICPULayer {
public:
    CPUConvolution(deepworks::LayerInfo&& info);
    virtual void forward(const std::vector<deepworks::Tensor>& inputs,
                               std::vector<deepworks::Tensor>& outputs) override;

    virtual void updateGradients(const std::vector<Tensor>& inputs,
                                 const std::vector<Tensor>& grad_outputs) override;

    virtual void backward(const std::vector<Tensor>& inputs,
                          const std::vector<Tensor>& outputs,
                          const std::vector<Tensor>& grad_outputs,
                                std::vector<Tensor>& grad_inputs) override;

private:
    void validate(const std::vector<deepworks::Tensor>& inputs,
                  const std::vector<deepworks::Tensor>& outputs);

    Tensor im2col_buf;
};

} // namespace cpu
} // namespace deepworks
