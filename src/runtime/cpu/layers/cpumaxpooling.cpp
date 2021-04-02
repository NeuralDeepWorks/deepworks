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
