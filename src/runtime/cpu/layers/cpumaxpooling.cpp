#include "runtime/cpu/layers/cpumaxpooling.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

void deepworks::cpu::CPUMaxPooling::validate(const std::vector<deepworks::Tensor>& inputs,
                                             const std::vector<deepworks::Tensor>& outputs) {
    const auto& kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    const auto& padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    const auto& stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    DeepWorks_Assert(kernel.size() == 2u && kernel[0] > 0 && kernel[1] > 0);
    DeepWorks_Assert(padding.size() == 2u && padding[0] >= 0 && padding[1] >= 0);
    DeepWorks_Assert(stride.size() == 2u && stride[0] > 0 && stride[1] > 0);
}

void deepworks::cpu::CPUMaxPooling::forward(const std::vector<deepworks::Tensor>& inputs,
                                                  std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
          auto& output = outputs.front();

    const auto& kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    const auto& padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    const auto& stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    // FIXME: Move it to init method
    if (max_indices.shape() != output.shape()) {
        max_indices.allocate(output.shape());
    }

    int h_out = output.shape()[2];
    int w_out = output.shape()[3];

    int rows = kernel[0] * kernel[1];
    int cols = h_out * w_out;

    // FIXME: Move it to init method
    if (im2col_buf.total() != rows * cols) {
        im2col_buf.allocate({rows * cols});
    }

    CPUMaxPoolingForward(input, max_indices, im2col_buf, output, kernel, padding, stride);
}

void deepworks::cpu::CPUMaxPooling::backward(const std::vector<deepworks::Tensor>& /* inputs */,
                                             const std::vector<deepworks::Tensor>& /* outputs */,
                                             const std::vector<deepworks::Tensor>& grad_outputs,
                                                   std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& grad_output = grad_outputs.front();
          auto& grad_input  = grad_inputs.front();

    DeepWorks_Assert(max_indices.shape() == grad_output.shape() && "Call backward without forward");

    const auto& kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    const auto& padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    const auto& stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    deepworks::CPUMaxPoolingInputGrad(grad_output, max_indices, im2col_buf, grad_input, kernel, padding, stride);
}
