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
    void CPUReLUBackward(const float* in, const float* grad_output,
                         float* grad_input, size_t batch_size, size_t features);

    float CPUCrossEntropyLossForward(const Tensor& predictions,
                                     const Tensor& target);
    void CPUCrossEntropyLossBackward(const Tensor& predictions,
                                     const Tensor& target,
                                     Tensor& grad_output);

    void SGDStep(Parameters& params, float learning_rate);

    void CPUBatchNorm1DForward(const Tensor& input, Tensor& output,
                               Tensor& input_centered, Tensor& std,
                               Tensor& running_mean, Tensor& running_var,
                               bool is_training, float eps, float alpha,
                               const Tensor& gamma, const Tensor& beta);
    void CPUBatchNorm1DBackward(const Tensor& input_centered, const Tensor& std,
                                const Tensor& grad_output, Tensor& grad_input,
                                const Tensor& gamma, Tensor& gamma_grad, Tensor& betta_grad);

    void MatMul(const float* in1, const float* in2, float* out, size_t m, size_t n, size_t l);
    std::vector<float> Transpose(const float* in, size_t rows, size_t cols);

} // namespace reference
} // namespace deepworks
