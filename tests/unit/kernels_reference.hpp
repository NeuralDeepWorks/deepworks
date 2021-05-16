#pragma once

#include <vector>
#include <deepworks/tensor.hpp>
#include <deepworks/parameter.hpp>

namespace deepworks {
namespace reference {
    void CPULinearForward(const float* X, const float* W, float* result,
                          size_t batch_size, size_t in_features, size_t out_features);
    void CPULinearAddBias(const float* b, float* result, size_t batch_size, size_t out_features);

    void CPULinearBackward(const float* input, const float* W, const float* grad_output, float* dW, float* grad_input,
                           size_t batch_size, size_t in_features, size_t out_features);
    void CPULinearBiasBackward(const float* grad_output, float* db, size_t batch_size, size_t out_features);

    void CPUSoftmaxForward(const float* X, float* result, size_t batch_size, size_t in_features);
    void CPUSoftmaxBackward(const float* grad_output, const float* output, float* grad_input,
                            size_t batch_size, size_t in_features);

    void CPUReLUForward(const float* in, float* out, size_t size);
    void CPUReLUBackward(const float* out, const float* grad_output,
                         float* grad_input, size_t size);

    void CPUELUForward(const Tensor& in, Tensor& out, float alpha);
    void CPUELUBackward(const Tensor& in, const Tensor& grad_output,
                              Tensor& grad_input, float alpha);

    void CPULeakyReLUForward(const Tensor& in, Tensor& out, float alpha);
    void CPULeakyReLUBackward(const Tensor& in, const Tensor& grad_output,
                                    Tensor& grad_input, float alpha);

    void CPUSigmoidForward(const Tensor& in, Tensor& out);
    void CPUSigmoidBackward(const Tensor& in, const Tensor& grad_output, Tensor& grad_input);

    float CPUCrossEntropyLossForward(const Tensor& predictions,
                                     const Tensor& target);
    void CPUCrossEntropyLossBackward(const Tensor& predictions,
                                     const Tensor& target,
                                     Tensor& grad_output);

    void SGDStep(ParamMap& params, float learning_rate);
    void SGDMomentumStep(ParamMap& params, TensorMap& velocities, float learning_rate, float gamma);

    void AdamStep(ParamMap& params, TensorMap& moving_mean, TensorMap& moving_variance,
                  float learning_rate, std::array<float, 2>& betas, float epsilon, size_t n_iterations);

    void CPUBatchNorm1DForward(const Tensor& input, Tensor& output,
                               Tensor& input_centered, Tensor& std,
                               Tensor& running_mean, Tensor& running_var,
                               bool is_training, float eps, float alpha,
                               const Tensor& gamma, const Tensor& beta);

    void CPUBatchNorm1DBackward(const Tensor& input_centered, const Tensor& std,
                                const Tensor& grad_output, Tensor& grad_input,
                                const Tensor& gamma, Tensor& gamma_grad, Tensor& betta_grad);

    void CPUBatchNorm2DForward(const Tensor& input,
                               Tensor& output,
                               const Tensor& gamma,
                               const Tensor& beta,
                               Tensor& running_mean,
                               Tensor& running_var,
                               Tensor& input_centered,
                               Tensor& std,
                               bool  is_train,
                               float alpha,
                               float eps);

    void CPUBatchNorm2DBackward(const Tensor& input_centered, const Tensor& std,
                                const Tensor& grad_output, Tensor& grad_input,
                                const Tensor& gamma, Tensor& gamma_grad, Tensor& betta_grad);

    void MatMul(const float* in1, const float* in2, float* out, size_t m, size_t n, size_t l);
    std::vector<float> Transpose(const float* in, size_t rows, size_t cols);

    void CPUMaxPooling2DForward(const deepworks::Tensor& input, deepworks::Tensor& output, const std::array<int, 2>& kernel,
                                const std::array<int, 2>& padding, const std::array<int, 2>& stride);

    void CPUMaxPooling2DBackward(const deepworks::Tensor& input, const deepworks::Tensor& grad_output, deepworks::Tensor& grad_input,
                                  const std::array<int, 2>& kernel, const std::array<int, 2>& padding, const std::array<int, 2>& stride);

    void CPUConvolution2DForward(const deepworks::Tensor& input, const deepworks::Tensor& weights, const deepworks::Tensor& bias,
                                 deepworks::Tensor& output, const std::array<int, 2>& kernel, const std::array<int, 2>& padding,
                                 const std::array<int, 2>& stride);

    void CPUConvolution2DBackward(const deepworks::Tensor& input, const deepworks::Tensor& grad_output, const deepworks::Tensor& weights,
                                  const deepworks::Tensor& bias, deepworks::Tensor& grad_weights, deepworks::Tensor& grad_bias,
                                  deepworks::Tensor& grad_input, const std::array<int, 2>& kernel, const std::array<int, 2>& padding,
                                  const std::array<int, 2>& stride);

    void CPUDropoutForward(const Tensor& in, const Tensor& mask, Tensor& out, float p, bool is_train);

    void CPUDropoutBackward(const Tensor& mask, const Tensor& grad_output, Tensor& grad_input, float p);

    void CPUGlobalAvgPoolingForward(const Tensor& input, Tensor& output);

    void CPUGlobalAvgPoolingBackward(const Tensor& grad_output, Tensor& grad_input);

    void CPUAddForward(const Tensor& input0, const Tensor& input1, Tensor& output);
    void CPUAddBackward(const Tensor& grad_output, Tensor& grad_input0, Tensor& grad_input1);
} // namespace reference
} // namespace deepworks
