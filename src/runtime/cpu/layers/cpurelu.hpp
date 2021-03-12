#pragma once

#include <deepworks/tensor.hpp>

#include "runtime/cpu/layers/cpulayer.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

#include "util/assert.hpp"

namespace deepworks {
namespace cpu {

class CPUReLU : public ICPULayer {
public:
    CPUReLU(deepworks::LayerInfo info);
    virtual void forward(const std::vector<Tensor>& inputs,
                               std::vector<Tensor>& outputs) override;
};

CPUReLU::CPUReLU(deepworks::LayerInfo info) : ICPULayer(info) { }

void CPUReLU::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    DeepWorks_Assert(inputs.size() == 1u  && "ReLU takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "ReLU produce only one output");

    const auto& input  = inputs.front();
          auto& output = outputs.front();

    // FIXME: JIZZ remove it !!!!
    //auto* raw = const_cast<float*>(input.data());
    //auto&& in_shape = input.shape();
    //MatrixMapper in_mapper(raw, in_shape[0], in_shape[1]);

    //auto&& out_shape = output.shape();
    //MatrixMapper out_mapper(output.data(), out_shape[0], out_shape[1]);

    //deepworks::cpu::kernels::CPUReLUForward(in_mapper, out_mapper);
}

} // namespace cpu
} // namespace deepworks
