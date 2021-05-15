#include "runtime/cpu/layers/cpuadd.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

void deepworks::cpu::CPUAdd::forward(const std::vector<deepworks::Tensor>& inputs,
                                           std::vector<deepworks::Tensor>& outputs) {
    const auto& lhs = inputs.at(0);
    const auto& rhs = inputs.at(1);
          auto& out = outputs.at(0);

    CPUAddForward(lhs, rhs, out);
}

void deepworks::cpu::CPUAdd::backward(const  std::vector<deepworks::Tensor>& inputs,
                                      const  std::vector<deepworks::Tensor>& /* outputs */,
                                      const  std::vector<deepworks::Tensor>& grad_outputs,
                                             std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& go   = grad_outputs.at(0);
          auto& gi_0 = grad_inputs.at(0);
          auto& gi_1 = grad_inputs.at(1);

    for (int i = 0; i < go.total(); ++i) {
        gi_0.data()[i] += go.data()[i];
    }

    for (int i = 0; i < go.total(); ++i) {
        gi_1.data()[i] += go.data()[i];
    }
}
