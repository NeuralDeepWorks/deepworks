#include "runtime/cpu/layers/cpulinear.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

deepworks::cpu::CPULinear::CPULinear(LayerInfo&& info)
    : ICPULayer(std::move(info)),
      m_W(m_info.params().at("weight").data()),
      m_b(m_info.params().at("bias").data()),
      m_gradW(m_info.params().at("weight").grad()),
      m_gradb(m_info.params().at("bias").grad()) {
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

    const auto& w_shape   = m_W.shape();
    const auto& b_shape   = m_b.shape();
    const auto& in_shape  = input.shape();
    const auto& out_shape = output.shape();

    deepworks::CPULinearForward({input.data() , in_shape[0] , in_shape[1]},
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

    deepworks::CPULinearInputGrad({grad_output.data(), grad_output_shape[0], grad_output_shape[1]},
                                  {m_W.data(), w_shape[0], w_shape[1]},
                                  {grad_input.data(), grad_input_shape[0], grad_input_shape[1]});
}

void deepworks::cpu::CPULinear::updateGradients(const std::vector<Tensor>& inputs,
                                                const std::vector<Tensor>& grad_outputs) {
    const auto& w_grad_shape      = m_gradW.shape();
    const auto& b_grad_shape      = m_gradb.shape();
    const auto& grad_output       = grad_outputs.front();
    const auto& grad_output_shape = grad_output.shape();
    const auto& input             = inputs.front();
    const auto& input_shape       = input.shape();

    deepworks::CPULinearWeightGrad({input.data(), input_shape[0], input_shape[1]},
                                   {grad_output.data(), grad_output_shape[0], grad_output_shape[1]},
                                   {m_gradW.data(), w_grad_shape[0], w_grad_shape[1]});

    deepworks::CPULinearBiasGrad({grad_output.data(), grad_output_shape[0], grad_output_shape[1]},
                                 {m_gradb.data(), b_grad_shape[0]});
}
