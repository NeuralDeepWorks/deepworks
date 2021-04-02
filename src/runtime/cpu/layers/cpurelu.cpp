#include "runtime/cpu/layers/cpurelu.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

void deepworks::cpu::CPUReLU::validate(const std::vector<deepworks::Tensor>& inputs,
                                       const std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "ReLU takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "ReLU produce only one output");

    DeepWorks_Assert(inputs.front().shape() == outputs.front().shape()
                     && "ReLU input and output must have the same shape");
}

void deepworks::cpu::CPUReLU::forward(const std::vector<deepworks::Tensor>& inputs,
                                            std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
          auto& output = outputs.front();

    deepworks::CPUReLUForward({input.data() , static_cast<int>(input.total())},
                              {output.data(), static_cast<int>(output.total())});
}

void deepworks::cpu::CPUReLU::backward(const std::vector<deepworks::Tensor>& inputs,
                                       const std::vector<deepworks::Tensor>& /* outputs */,
                                       const std::vector<deepworks::Tensor>& grad_outputs,
                                             std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& input             = inputs.front();
    const auto& grad_output       = grad_outputs.front();
          auto& grad_input        = grad_inputs.front();

    deepworks::CPUReLUInputGrad({input.data()      , static_cast<int>(input.total())},
                                {grad_output.data(), static_cast<int>(grad_output.total())},
                                {grad_input.data() , static_cast<int>(grad_input.total())});
}
