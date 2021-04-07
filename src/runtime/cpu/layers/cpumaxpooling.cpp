#include "runtime/cpu/layers/cpumaxpooling.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

deepworks::cpu::CPUMaxPooling::CPUMaxPooling(deepworks::LayerInfo&& info)
    : deepworks::cpu::ICPULayer(std::move(info)) {
}

void deepworks::cpu::CPUMaxPooling::validate(const std::vector<deepworks::Tensor>& inputs,
                                             const std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "MaxPooling takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "MaxPooling produce only one output");

    DeepWorks_Assert(inputs.front().shape().size()  == 4u && "MaxPooling input must be 4D");
    DeepWorks_Assert(outputs.front().shape().size() == 4u && "MaxPooling output must be 4D");

    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    DeepWorks_Assert(kernel.size() == 2u && kernel[0] > 0 && kernel[1] > 0);
    DeepWorks_Assert(padding.size() == 2u && padding[0] >= 0 && padding[1] >= 0);
    DeepWorks_Assert(stride.size() == 2u && stride[0] > 0 && stride[1] > 0);
}

void deepworks::cpu::CPUMaxPooling::forward(const std::vector<deepworks::Tensor>& inputs,
                                                  std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
          auto& output = outputs.front();

    // FIXME: Move it to init method
    if (max_indices.shape() != output.shape()) {
        max_indices.allocate(output.shape());
    }

    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();
    CPUMaxPoolingForward(input, max_indices, output, kernel, padding, stride);
}

void deepworks::cpu::CPUMaxPooling::backward(const std::vector<deepworks::Tensor>& /* inputs */,
                                             const std::vector<deepworks::Tensor>& /* outputs */,
                                             const std::vector<deepworks::Tensor>& grad_outputs,
                                                   std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& grad_output = grad_outputs.front();
          auto& grad_input  = grad_inputs.front();

    DeepWorks_Assert(max_indices.shape() == grad_output.shape() && "Call backward without forward");

    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    deepworks::CPUMaxPoolingInputGrad(grad_output, max_indices, grad_input, kernel, padding, stride);
}
