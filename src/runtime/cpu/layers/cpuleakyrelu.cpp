#include "runtime/cpu/layers/cpuleakyrelu.hpp"

void deepworks::cpu::LeakyReLU::validate(const std::vector<deepworks::Tensor>& inputs,
                                         const std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "LeakyReLU takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "LeakyReLU produce only one output");

    DeepWorks_Assert(inputs.front().shape() == outputs.front().shape()
                     && "LeakyReLU input and output must have the same shape");
}

void deepworks::cpu::LeakyReLU::forward(const std::vector<deepworks::Tensor>& inputs,
                                        std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
    auto& output = outputs.front();

    float alpha = m_info.impl().attrs["alpha"].get<float>();

    deepworks::CPULeakyReLUForward({input.data(), static_cast<int>(input.total())},
                                   {output.data(), static_cast<int>(output.total())}, alpha);
}

void deepworks::cpu::LeakyReLU::backward(const std::vector<deepworks::Tensor>& inputs,
                                         const std::vector<deepworks::Tensor>& /* outputs */,
                                         const std::vector<deepworks::Tensor>& grad_outputs,
                                         std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& input       = inputs.front();
    const auto& grad_output = grad_outputs.front();
    auto& grad_input        = grad_inputs.front();

    float alpha = m_info.impl().attrs["alpha"].get<float>();

    deepworks::CPULeakyReLUInputGrad({input.data(), static_cast<int>(input.total())},
                                     {grad_output.data(), static_cast<int>(grad_output.total())},
                                     {grad_input.data() , static_cast<int>(grad_input.total())}, alpha);
}
