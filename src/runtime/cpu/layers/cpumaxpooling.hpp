#pragma once

#include <deepworks/tensor.hpp>
#include "runtime/cpu/layers/cpulayer.hpp"
#include "util/assert.hpp"

namespace deepworks {
namespace cpu {

class CPUMaxPooling : public ICPULayer {
public:
    CPUMaxPooling(deepworks::LayerInfo&& info);
    virtual void forward(const std::vector<deepworks::Tensor>& inputs,
                               std::vector<deepworks::Tensor>& outputs) override;
private:
    void validate(const std::vector<deepworks::Tensor>& inputs,
                  const std::vector<deepworks::Tensor>& outputs);

    Tensor max_indices;
};

} // namespace cpu
} // namespace deepworks
