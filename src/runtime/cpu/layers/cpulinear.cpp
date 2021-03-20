#include "runtime/cpu/layers/cpulinear.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

deepworks::cpu::CPULinear::CPULinear(deepworks::LayerInfo&& info)
    : deepworks::cpu::ICPULayer(std::move(info)) {
}

void deepworks::cpu::CPULinear::validate(const std::vector<deepworks::Tensor>& inputs,
                                         const std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "Linear takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "Linear produce only one output");

    DeepWorks_Assert(inputs.front().shape().size()  == 2u && "Linear input must be 2D");
    DeepWorks_Assert(outputs.front().shape().size() == 2u && "Linear output must be 2D");
}

void deepworks::cpu::CPULinear::forward(const std::vector<deepworks::Tensor>& inputs,
                                              std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
          auto& output = outputs.front();
    const auto& weight = m_info.params()[0].data();
    const auto& bias   = m_info.params()[1].data();

    const auto& w_shape   = weight.shape();
    const auto& b_shape   = bias.shape();
    const auto& in_shape  = input.shape();
    const auto& out_shape = output.shape();

    deepworks::CPULinearForward({input.data() , in_shape[0] , in_shape[1]},
                                {weight.data(), w_shape[0]  , w_shape[1]},
                                {output.data(), out_shape[0], out_shape[1]});

    deepworks::CPULinearAddBias({bias.data(), b_shape[0]},
                                {output.data(), out_shape[0], out_shape[1]});
}
