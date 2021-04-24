#include <numeric>

#include "runtime/cpu/layers/cpulinear.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

deepworks::cpu::CPULinear::CPULinear(LayerInfo&& info)
    : ICPULayer(std::move(info)),
      m_W(m_info.params()[0].data()),
      m_b(m_info.params()[1].data()),
      m_gradW(m_info.params()[0].grad()),
      m_gradb(m_info.params()[1].grad()) {
}

void deepworks::cpu::CPULinear::validate(const std::vector<deepworks::Tensor>& inputs,
                                         const std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "Linear takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "Linear produce only one output");

    DeepWorks_Assert(inputs.front().shape().size()  != 1u && "Linear input mustn't be 1D");
    DeepWorks_Assert(outputs.front().shape().size() != 1u && "Linear output mustn't be 1D");
}

void deepworks::cpu::CPULinear::forward(const std::vector<deepworks::Tensor>& inputs,
                                              std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
          auto& output = outputs.front();

    const auto& w_shape   = m_W.shape();
    const auto& b_shape   = m_b.shape();
    const auto& in_shape  = input.shape();
    const auto& out_shape = output.shape();

    auto second_shape = std::accumulate(in_shape.begin() + 1, in_shape.end(), 1, std::multiplies<int>());

    deepworks::CPULinearForward({input.data() , in_shape[0] , second_shape},
                                {m_W.data()   , w_shape[0]  , w_shape[1]},
                                {output.data(), out_shape[0], out_shape[1]});

    deepworks::CPULinearAddBias({output.data(), out_shape[0], out_shape[1]},
                                {m_b.data(), b_shape[0]},
                                {output.data(), out_shape[0], out_shape[1]});
}

void deepworks::cpu::CPULinear::backward(const std::vector<deepworks::Tensor>& inputs,
                                         const std::vector<deepworks::Tensor>& /* outputs */,
                                         const std::vector<deepworks::Tensor>& grad_outputs,
                                               std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& w_shape           = m_W.shape();
    const auto& b_shape           = m_b.shape();
    const auto& grad_output       = grad_outputs.front();
    const auto& grad_output_shape = grad_output.shape();
    const auto& input             = inputs.front();
    const auto& input_shape       = input.shape();
          auto& grad_input        = grad_inputs.front();
          auto& grad_input_shape  = grad_input.shape();

    auto second_shape = std::accumulate(input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<int>());

    deepworks::CPULinearInputGrad({grad_output.data(), grad_output_shape[0], grad_output_shape[1]},
                                  {m_W.data(), w_shape[0], w_shape[1]},
                                  {grad_input.data(), grad_input_shape[0], second_shape});
}

void deepworks::cpu::CPULinear::updateGradients(const std::vector<Tensor>& inputs,
                                                const std::vector<Tensor>& grad_outputs) {
    const auto& w_grad_shape      = m_gradW.shape();
    const auto& b_grad_shape      = m_gradb.shape();
    const auto& grad_output       = grad_outputs.front();
    const auto& grad_output_shape = grad_output.shape();
    const auto& input             = inputs.front();
    const auto& input_shape       = input.shape();

    auto second_shape = std::accumulate(input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<int>());

    deepworks::CPULinearWeightGrad({input.data(), input_shape[0], second_shape},
                                   {grad_output.data(), grad_output_shape[0], grad_output_shape[1]},
                                   {m_gradW.data(), w_grad_shape[0], w_grad_shape[1]});

    deepworks::CPULinearBiasGrad({grad_output.data(), grad_output_shape[0], grad_output_shape[1]},
                                 {m_gradb.data(), b_grad_shape[0]});
}
