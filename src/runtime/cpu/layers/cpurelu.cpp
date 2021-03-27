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

    const auto& in_shape  = input.shape();
    const auto& out_shape = output.shape();

    deepworks::CPUReLUForward({input.data() , in_shape[0] , in_shape[1]},
                              {output.data(), out_shape[0], out_shape[1]});
}

void deepworks::cpu::CPUReLU::backward(const std::vector<deepworks::Tensor>& inputs,
                                       const std::vector<deepworks::Tensor>& /* outputs */,
                                       const std::vector<deepworks::Tensor>& grad_outputs,
                                             std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& input             = inputs.front();
    const auto& in_shape          = input.shape();
    const auto& grad_output       = grad_outputs.front();
    const auto& grad_output_shape = grad_output.shape();
          auto& grad_input        = grad_inputs.front();
    const auto& grad_input_shape  = grad_input.shape();

    deepworks::CPUReLUInputGrad({input.data() , in_shape[0] , in_shape[1]},
                                {grad_output.data(), grad_output_shape[0], grad_output_shape[1]},
                                {grad_input.data(), grad_input_shape[0], grad_input_shape[1]});
}
