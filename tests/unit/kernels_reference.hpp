#include <deepworks/tensor.hpp>

namespace deepworks {
namespace reference {

    void CPULinearForward(const float* X, const float* W, float* result,
                          size_t batch_size, size_t in_features, size_t out_features);
    void CPULinearAddBias(const float* b, float* result, size_t batch_size, size_t out_features);

    void CPULinearBackward(const float* input, const float* W, const float* dx, float* dW, float* grad_output,
                           size_t batch_size, size_t in_features, size_t out_features);
    void CPULinearBiasBackward(const float* dx, float* db, size_t batch_size, size_t out_features);

    void CPUSoftmaxForward(const float* X, float* result, size_t batch_size, size_t in_features);
    void CPUSoftmaxBackward(const float* dx, const float* output, float* grad_output,
                            size_t batch_size, size_t in_features);

    void CPUReLUForward(const float* in, float* out, size_t size);
    void CPUReLUBackward(const float* dx, const float* output, float* grad_output, size_t size);

    float CPUCrossEntropyLossForward(const deepworks::Tensor& predictions,
                                     const deepworks::Tensor& target);
    void CPUCrossEntropyLossBackward(const deepworks::Tensor& predictions,
                                     const deepworks::Tensor& target,
                                     deepworks::Tensor& grad_output);

    void CPUBatchNorm1DForward(const float* input, float* output,
                               float* running_mean, float* running_var,
                               bool isTraining, float eps, float alpha,
                               const float* gamma, const float* beta,
                               size_t rows, size_t cols);
    void CPUBatchNorm1DBackward(float* input_centered, float* std,
                                float* grad_output, float* grad_input,
                                const float* gamma, float* gamma_grad,
                                float* betta_grad, size_t rows, size_t cols);

    void Multiply(const float* in1, const float* in2, float* out, size_t m, size_t n, size_t l);
    std::vector<float> Transpose(const float* in, size_t rows, size_t cols);

} // namespace reference
} // namespace deepworks
