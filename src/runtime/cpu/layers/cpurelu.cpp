#include "runtime/cpu/layers/cpurelu.hpp"

deepworks::cpu::CPUReLU::CPUReLU(deepworks::LayerInfo info)
    : deepworks::cpu::ICPULayer(info) {
}

void deepworks::cpu::CPUReLU::validate(const std::vector<deepworks::Tensor>& inputs,
                                             std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "ReLU takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "ReLU produce only one output");

    DeepWorks_Assert(inputs.front().shape().size()  == 2u && "ReLU input must be 2D");
    DeepWorks_Assert(outputs.front().shape().size() == 2u && "ReLU output must be 2D");
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
