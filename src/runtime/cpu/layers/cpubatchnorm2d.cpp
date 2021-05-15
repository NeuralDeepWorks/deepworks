#include "runtime/cpu/layers/cpubatchnorm2d.hpp"

#include <deepworks/initializers.hpp>
#include <deepworks/utils/utils.hpp>

deepworks::cpu::CPUBatchNorm2D::CPUBatchNorm2D(deepworks::LayerInfo&& info)
    : deepworks::cpu::ICPULayer(std::move(info)),
      m_gamma(m_info.params().at("gamma").data()),
      m_beta(m_info.params().at("beta").data()),
      m_grad_gamma(m_info.params().at("gamma").grad()),
      m_grad_beta(m_info.params().at("beta").grad()),
      m_std(deepworks::Tensor::zeros({m_gamma.shape()[0]})),
      m_running_mean(m_info.buffers().at("running_mean")),
      m_running_var(m_info.buffers().at("running_var")) {
}

void deepworks::cpu::CPUBatchNorm2D::validate(const std::vector<deepworks::Tensor>& inputs,
                                          std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(inputs.size()  == 1u && "BatchNorm2D takes only one input");
    DeepWorks_Assert(outputs.size() == 1u && "BatchNorm2D produce only one output");

    DeepWorks_Assert(inputs.front().shape().size()  == 4u && "BatchNorm2D input must be 4D");
    DeepWorks_Assert(outputs.front().shape().size() == 4u && "BatchNorm2D output must be 4D");
}

void deepworks::cpu::CPUBatchNorm2D::forward(const std::vector<deepworks::Tensor>& inputs,
                                             std::vector<deepworks::Tensor>& outputs) {
    validate(inputs, outputs);

    const auto& input  = inputs.front();
    auto& output = outputs.front();

    const auto& in_shape  = input.shape();

    // Check if model in train mode
    bool gamma_is_trainable = m_info.params().at("gamma").is_trainable();
    bool beta_is_trainable  = m_info.params().at("beta").is_trainable();

    DeepWorks_Assert(gamma_is_trainable == beta_is_trainable &&
                     "Gamma and beta should have same trainable mode");

    bool is_trainable = gamma_is_trainable;

    float eps   = m_info.impl().attrs["eps"].get<float>();
    float alpha = m_info.impl().attrs["alpha"].get<float>();

    int N = in_shape[0];
    int C = in_shape[1];
    int H = in_shape[2];
    int W = in_shape[3];

    if (nhwc_in.empty()) {
        nhwc_in  = Tensor({N, H, W, C});
        nhwc_out = Tensor({N, H, W, C});
    }

    if (m_input_centered.empty()) {
        m_input_centered = Tensor({N * H * W, C});
    }

    deepworks::utils::NCHW2NHWC(input, nhwc_in);

    deepworks::CPUBatchNorm1DForward({nhwc_in.data() , N * H * W, C},
                                     {nhwc_out.data(), N * H * W, C},
                                     {m_input_centered.data(), N * H * W , C},
                                     {m_std.data(), C},
                                     {m_running_mean.data(), C},
                                     {m_running_var.data(), C},
                                     is_trainable, eps, alpha,
                                     {m_gamma.data(), C},
                                     {m_beta.data(), C});

    deepworks::utils::NHWC2NCHW(nhwc_out, output);
}

void deepworks::cpu::CPUBatchNorm2D::backward(const std::vector<deepworks::Tensor>& /* inputs */,
                                              const std::vector<deepworks::Tensor>& /* outputs */,
                                              const std::vector<deepworks::Tensor>& grad_outputs,
                                              std::vector<deepworks::Tensor>& grad_inputs) {
    const auto& input_centered_shape  = m_input_centered.shape();
    const auto& grad_output           = grad_outputs.front();
    const auto& grad_output_shape     = grad_output.shape();
          auto& grad_input            = grad_inputs.front();

    int C = grad_output_shape[1];
    deepworks::utils::NCHW2NHWC(grad_output, nhwc_out);

    // NB: CPUBatchNorm1DInputGrad expects that input_grad is already initiliazed.
    deepworks::initializer::zeros(nhwc_in);
    deepworks::CPUBatchNorm1DInputGrad({m_input_centered.data(), static_cast<int>(grad_output.total()) / C, C},
                                       {m_std.data(),C},
                                       {nhwc_out.data(), static_cast<int>(grad_output.total()) / C, C},
                                       {nhwc_in.data(), static_cast<int>(nhwc_in.total()) / C, C},
                                       {m_gamma.data(), C});

    deepworks::utils::NHWC2NCHW(nhwc_in, grad_input);
}

void deepworks::cpu::CPUBatchNorm2D::updateGradients(const std::vector<Tensor>& /* inputs */,
                                                     const std::vector<Tensor>& grad_outputs) {
    const auto& input_centered_shape  = m_input_centered.shape();
    const auto& grad_output           = grad_outputs.front();

    int C = grad_output.shape()[1];
    deepworks::utils::NCHW2NHWC(grad_output, nhwc_out);

    deepworks::CPUBatchNorm1DParamGrad({m_input_centered.data(), static_cast<int>(grad_output.total()) / C, C},
                                       {m_std.data(), C},
                                       {nhwc_out.data(), static_cast<int>(grad_output.total()) / C, C},
                                       {m_grad_gamma.data(), C},
                                       {m_grad_beta.data(), C});
}
