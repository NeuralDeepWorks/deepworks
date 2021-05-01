#include "runtime/cpu/layers/cpudropout.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

deepworks::cpu::CPUDropout::CPUDropout(deepworks::LayerInfo&& info)
    : deepworks::cpu::ICPULayer(std::move(info)),
      m_mask(m_info.params().at("mask").data()) {
}

void deepworks::cpu::CPUDropout::validate(const std::vector<deepworks::Tensor>& inputs,
                                       const std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "Dropout takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "Dropout produce only one output");

    DeepWorks_Assert(inputs.front().shape() == outputs.front().shape()
                     && "Dropout input and output must have the same shape");
}

void deepworks::cpu::CPUDropout::forward(const std::vector<deepworks::Tensor>& inputs,
                                               std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
          auto& output = outputs.front();

    if (m_info.params().at("mask").is_trainable()) {
        float p = m_info.impl().attrs["p"].get<float>();
        deepworks::CPUDropoutForward(input, m_mask, output, p);
    } else {
        input.copyTo(output);
    }
}

void deepworks::cpu::CPUDropout::backward(const std::vector<deepworks::Tensor>& /* inputs */,
                                          const std::vector<deepworks::Tensor>& /* outputs */,
                                          const std::vector<deepworks::Tensor>& grad_outputs,
                                                std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& grad_output = grad_outputs.front();
          auto& grad_input  = grad_inputs.front();

    float p = m_info.impl().attrs["p"].get<float>();
    deepworks::CPUDropoutInputGrad(m_mask, grad_output, grad_input, p);
}
