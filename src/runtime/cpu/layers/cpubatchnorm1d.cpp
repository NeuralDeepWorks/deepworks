#include "runtime/cpu/layers/cpubatchnorm1d.hpp"

#include <deepworks/initializers.hpp>
#include <deepworks/utils/utils.hpp>

deepworks::cpu::CPUBatchNorm1D::CPUBatchNorm1D(deepworks::LayerInfo&& info)
    : deepworks::cpu::ICPULayer(std::move(info)),
      m_gamma(m_info.params().at("gamma").data()),
      m_beta(m_info.params().at("beta").data()),
      m_grad_gamma(m_info.params().at("gamma").grad()),
      m_grad_beta(m_info.params().at("beta").grad()),
      m_std(deepworks::Tensor::zeros({m_gamma.shape()[0]})),
      m_running_mean(m_info.buffers().at("running_mean")),
      m_running_var(m_info.buffers().at("running_var")) {
}

void deepworks::cpu::CPUBatchNorm1D::validate(const std::vector<deepworks::Tensor>& inputs,
                                          std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "BatchNorm1D takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "BatchNorm1D produce only one output");

    DeepWorks_Assert(inputs.front().shape().size()  == 2u && "BatchNorm1D input must be 2D");
    DeepWorks_Assert(outputs.front().shape().size() == 2u && "BatchNorm1D output must be 2D");
}

void deepworks::cpu::CPUBatchNorm1D::forward(const std::vector<deepworks::Tensor>& inputs,
                                                   std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
    auto& output = outputs.front();

    const auto& in_shape  = input.shape();
    const auto& out_shape = output.shape();

    // If change batch_size need to create new m_input_centered tensor
    if (m_input_centered.shape() != input.shape()) {
        m_input_centered = Tensor(in_shape);
    }

    // Check if model in train mode
    bool gamma_is_trainable = m_info.params().at("gamma").is_trainable();
    bool beta_is_trainable  = m_info.params().at("beta").is_trainable();

    bool is_trainable = gamma_is_trainable && beta_is_trainable;

    DeepWorks_Assert(gamma_is_trainable == beta_is_trainable &&
                     "Gamma and beta should have same trainable mode");

    const auto& gamma_shape           = m_gamma.shape();
    const auto& beta_shape            = m_beta.shape();
    const auto& input_centered_shape  = m_input_centered.shape();
    const auto& std_shape             = m_std.shape();
    const auto& running_mean_shape    = m_running_mean.shape();
    const auto& running_var_shape     = m_running_var.shape();

    float eps   = m_info.impl().attrs["eps"].get<float>();
    float alpha = m_info.impl().attrs["alpha"].get<float>();

    deepworks::CPUBatchNorm1DForward({input.data() , in_shape[0] , in_shape[1]},
                                     {output.data(), out_shape[0], out_shape[1]},
                                     {m_input_centered.data(), input_centered_shape[0], input_centered_shape[1]},
                                     {m_std.data(), std_shape[0]},
                                     {m_running_mean.data(), running_mean_shape[0]},
                                     {m_running_var.data(), running_var_shape[0]},
                                     is_trainable, eps, alpha,
                                     {m_gamma.data(), gamma_shape[0]},
                                     {m_beta.data(), beta_shape[0]});
}

void deepworks::cpu::CPUBatchNorm1D::backward(const std::vector<deepworks::Tensor>& /* inputs */,
                                              const std::vector<deepworks::Tensor>& /* outputs */,
                                              const std::vector<deepworks::Tensor>& grad_outputs,
                                              std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& input_centered_shape  = m_input_centered.shape();
    const auto& std_shape             = m_std.shape();
    const auto& gamma_shape           = m_gamma.shape();
    const auto& grad_output           = grad_outputs.front();
    const auto& grad_output_shape     = grad_output.shape();
          auto& grad_input            = grad_inputs.front();
          auto& grad_input_shape      = grad_input.shape();

    deepworks::CPUBatchNorm1DInputGrad({m_input_centered.data(), input_centered_shape[0], input_centered_shape[1]},
                                       {m_std.data(), std_shape[0]},
                                       {grad_output.data(), grad_output_shape[0], grad_output_shape[1]},
                                       {grad_input.data(), grad_input_shape[0], grad_input_shape[1]},
                                       {m_gamma.data(), gamma_shape[0]});
}

void deepworks::cpu::CPUBatchNorm1D::updateGradients(const std::vector<Tensor>& /* inputs */,
                                                     const std::vector<Tensor>& grad_outputs) {

    const auto& input_centered_shape  = m_input_centered.shape();
    const auto& std_shape             = m_std.shape();
    const auto& gamma_grad_shape      = m_grad_gamma.shape();
    const auto& beta_grad_shape       = m_grad_beta.shape();
    const auto& grad_output           = grad_outputs.front();
    const auto& grad_output_shape     = grad_output.shape();

    deepworks::CPUBatchNorm1DParamGrad({m_input_centered.data(), input_centered_shape[0], input_centered_shape[1]},
                                       {m_std.data(), std_shape[0]},
                                       {grad_output.data(), grad_output_shape[0], grad_output_shape[1]},
                                       {m_grad_gamma.data(), gamma_grad_shape[0]},
                                       {m_grad_beta.data(), beta_grad_shape[0]});
}
