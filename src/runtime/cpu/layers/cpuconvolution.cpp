#include "runtime/cpu/layers/cpuconvolution.hpp"
#include "runtime/cpu/kernels/kernels.hpp"
#include <iostream>

deepworks::cpu::CPUConvolution::CPUConvolution(deepworks::LayerInfo&& info)
    : deepworks::cpu::ICPULayer(std::move(info)),
      grad_weights(m_info.params().at("weight").grad()),
      grad_bias(m_info.params().at("bias").grad()) {
}

void deepworks::cpu::CPUConvolution::validate(const std::vector<deepworks::Tensor>& inputs,
                                              const std::vector<deepworks::Tensor>& outputs) {
    const auto& kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    const auto& padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    const auto& stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    DeepWorks_Assert(kernel.size() == 2u && kernel[0] > 0 && kernel[1] > 0);
    DeepWorks_Assert(padding.size() == 2u && padding[0] >= 0 && padding[1] >= 0);
    DeepWorks_Assert(stride.size() == 2u && stride[0] > 0 && stride[1] > 0);
}

void deepworks::cpu::CPUConvolution::forward(const std::vector<deepworks::Tensor>& inputs,
                                                   std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input   = inputs.front();
          auto& output  = outputs.front();
    const auto& weights = m_info.params().at("weight").data();
    const auto& bias    = m_info.params().at("bias").data();

    const auto& kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    const auto& padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    const auto& stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    int batch = input.shape()[0];
    int c_in  = input.shape()[1];
    int h_out = output.shape()[2];
    int w_out = output.shape()[3];

    int rows = c_in * kernel[0] * kernel[1];
    int cols = h_out * w_out;

    // FIXME: Move it to init method
    if (im2col_buf.total() != batch * rows * cols) {
        im2col_buf.allocate({batch * rows * cols});
    }

    deepworks::CPUConvolutionalForward(input,
                                       weights,
                                       bias,
                                       output,
                                       im2col_buf,
                                       kernel,
                                       padding,
                                       stride);
}

void deepworks::cpu::CPUConvolution::backward(const std::vector<deepworks::Tensor>& /* inputs */,
                                              const std::vector<deepworks::Tensor>& /* outputs */,
                                              const std::vector<deepworks::Tensor>& grad_outputs,
                                                    std::vector<deepworks::Tensor>& grad_inputs) {
    DeepWorks_Assert(!im2col_buf.empty() && "Call backward without forward");

    const auto& grad_output = grad_outputs.front();
    const auto& weights     = m_info.params().at("weight").data();
          auto& grad_input  = grad_inputs.front();

    const auto& kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    const auto& padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    const auto& stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    deepworks::CPUConvolutionalInputGrad(grad_output,
                                         weights,
                                         im2col_buf,
                                         grad_input,
                                         kernel,
                                         padding,
                                         stride);

    //std::cout << "eigen : " << std::endl;
    //std::cout << grad_input << std::endl;
}

void deepworks::cpu::CPUConvolution::updateGradients(const std::vector<Tensor>& inputs,
                                                     const std::vector<Tensor>& grad_outputs) {
    DeepWorks_Assert(!im2col_buf.empty() && "Call backward without forward");

    const auto& grad_output  = grad_outputs.front();
    deepworks::CPUConvolutionalWeightsGrad(grad_output, im2col_buf, grad_weights);
    deepworks::CPUConvolutionalBiasGrad(grad_output, grad_bias);
}
