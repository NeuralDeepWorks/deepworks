#include "runtime/cpu/layers/cpusigmoid.hpp"

void deepworks::cpu::Sigmoid::validate(const std::vector<deepworks::Tensor>& inputs,
                                       const std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "Sigmoid takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "Sigmoid produce only one output");

    DeepWorks_Assert(inputs.front().shape() == outputs.front().shape()
                     && "Sigmoid input and output must have the same shape");
}

void deepworks::cpu::Sigmoid::forward(const std::vector<deepworks::Tensor>& inputs,
                                        std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
    auto& output = outputs.front();

    deepworks::CPUSigmoidForward({input.data(), static_cast<int>(input.total())},
                                 {output.data(), static_cast<int>(output.total())});
}

void deepworks::cpu::Sigmoid::backward(const std::vector<deepworks::Tensor>& /* inputs */,
                                       const std::vector<deepworks::Tensor>& outputs,
                                       const std::vector<deepworks::Tensor>& grad_outputs,
                                       std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& output      = outputs.front();
    const auto& grad_output = grad_outputs.front();
    auto& grad_input        = grad_inputs.front();

    deepworks::CPUSigmoidInputGrad({output.data(), static_cast<int>(output.total())},
                                   {grad_output.data(), static_cast<int>(grad_output.total())},
                                   {grad_input.data() , static_cast<int>(grad_input.total())});
}
