#include "runtime/cpu/layers/cpuconvolution.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

deepworks::cpu::CPUConvolution::CPUConvolution(deepworks::LayerInfo&& info)
    : deepworks::cpu::ICPULayer(std::move(info)) {
}

void deepworks::cpu::CPUConvolution::validate(const std::vector<deepworks::Tensor>& inputs,
                                                const std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "Convolutional takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "Convolutional produce only one output");

    DeepWorks_Assert(inputs.front().shape().size()  == 4u && "Convolutional input must be 4D");
    DeepWorks_Assert(outputs.front().shape().size() == 4u && "Convolutional output must be 4D");

    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    DeepWorks_Assert(kernel.size() == 2u && kernel[0] > 0 && kernel[1] > 0);
    DeepWorks_Assert(padding.size() == 2u && padding[0] >= 0 && padding[1] >= 0);
    DeepWorks_Assert(stride.size() == 2u && stride[0] > 0 && stride[1] > 0);
}

void deepworks::cpu::CPUConvolution::forward(const std::vector<deepworks::Tensor>& inputs,
                                                     std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input   = inputs.front();
          auto& output  = outputs.front();
    const auto& weights = m_info.params()[0].data();
    const auto& bias    = m_info.params()[1].data();

    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

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
