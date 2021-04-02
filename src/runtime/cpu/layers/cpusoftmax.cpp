#include "runtime/cpu/layers/cpusoftmax.hpp"

void deepworks::cpu::CPUSoftmax::validate(const std::vector<deepworks::Tensor>& inputs,
                                          const std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "Softmax takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "Softmax produce only one output");

    DeepWorks_Assert(inputs.front().shape().size()  == 2u && "Softmax input must be 2D");
    DeepWorks_Assert(outputs.front().shape().size() == 2u && "Softmax output must be 2D");
}

void deepworks::cpu::CPUSoftmax::forward(const std::vector<deepworks::Tensor>& inputs,
                                               std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
          auto& output = outputs.front();

    const auto& in_shape  = input.shape();
    const auto& out_shape = output.shape();
    deepworks::CPUSoftmaxForward({input.data() , in_shape[0] , in_shape[1]},
                                 {output.data(), out_shape[0], out_shape[1]});
}

void deepworks::cpu::CPUSoftmax::backward(const std::vector<deepworks::Tensor>& /* inputs */,
                                          const std::vector<deepworks::Tensor>& outputs,
                                          const std::vector<deepworks::Tensor>& grad_outputs,
                                                std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& grad_output = grad_outputs.front();
          auto& grad_input  = grad_inputs.front();
    const auto& output      = outputs.front();

    deepworks::CPUSoftmaxInputGrad({output.data()     , output.shape()[0]     , output.shape()[1]},
                                   {grad_output.data(), grad_output.shape()[0], grad_output.shape()[1]},
                                   {grad_input.data() , grad_input.shape()[0] , grad_input.shape()[1]});
}
