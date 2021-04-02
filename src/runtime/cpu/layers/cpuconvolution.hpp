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
private:
    void validate(const std::vector<deepworks::Tensor>& inputs,
                  const std::vector<deepworks::Tensor>& outputs);

    Tensor im2col_buf;
};

} // namespace cpu
} // namespace deepworks
