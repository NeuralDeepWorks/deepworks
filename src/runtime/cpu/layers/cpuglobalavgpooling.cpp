#include "runtime/cpu/layers/cpuglobalavgpooling.hpp"
#include "runtime/cpu/kernels/kernels.hpp"


void deepworks::cpu::CPUGlobalAvgPooling::forward(const std::vector<deepworks::Tensor>& inputs,
                                                  std::vector<deepworks::Tensor>& outputs) {
    const auto& input  = inputs.front();
          auto& output = outputs.front();

    deepworks::CPUGlobalAvgPoolingForward(input, output);
}

void deepworks::cpu::CPUGlobalAvgPooling::backward(const std::vector<deepworks::Tensor>& /* inputs */,
                                                   const std::vector<deepworks::Tensor>& /* outputs */,
                                                   const std::vector<deepworks::Tensor>& grad_outputs,
                                                         std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& grad_output = grad_outputs.front();
          auto& grad_input  = grad_inputs.front();

    deepworks::CPUGlobalAvgPoolingInputGrad(grad_output, grad_input);
}
