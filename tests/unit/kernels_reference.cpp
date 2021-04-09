#include "kernels_reference.hpp"

#include <deepworks/initializers.hpp>

#include <limits>
#include <cmath>

#include <algorithm>

namespace dw = deepworks;

void dw::reference::CPULinearForward(const float* X, const float* W, float* result,
                                     size_t batch_size, size_t in_features, size_t out_features) {
    auto WT = dw::reference::Transpose(W, out_features, in_features);
    dw::reference::MatMul(X, WT.data(), result, batch_size, in_features, out_features);
}

void dw::reference::CPULinearAddBias(const float* b, float* result, size_t batch_size, size_t out_features) {
    for (size_t sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        for (size_t i = 0; i < out_features; i++) {
            result[sample_idx * out_features + i] += b[i];
        }
    }
}

void dw::reference::CPULinearBackward(const float* input, const float* W, const float* grad_output,
                                      float* dW, float* grad_input, size_t batch_size,
                                      size_t in_features, size_t out_features) {
    // NB: Weight gradient
    auto grad_outputT = dw::reference::Transpose(grad_output, batch_size, out_features);
    dw::reference::MatMul(grad_outputT.data(), input, dW, out_features, batch_size, in_features);

    // NB: Input gradient
    dw::reference::MatMul(grad_output, W, grad_input, batch_size, out_features, in_features);
}

void dw::reference::CPULinearBiasBackward(const float* grad_output, float* db, size_t batch_size, size_t out_features) {
    for (size_t j = 0; j < out_features; j++) {
        float sum = 0.0;
        for (size_t i = 0; i < batch_size; i++) {
            sum += grad_output[i * out_features + j];
        }
        db[j] = sum;
    }
}

void dw::reference::CPUSoftmaxForward(const float* X, float* result, size_t batch_size, size_t in_features) {
    std::vector<float> rows_max(batch_size, std::numeric_limits<float>::min());

    // Find max feature for each sample
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < in_features; j++) {
            rows_max[i] = std::max(rows_max[i], X[i * in_features + j]);
        }
    }

    std::vector<float> exp_x(batch_size * in_features);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < in_features; j++) {
            exp_x[i * in_features + j] = std::exp(X[i * in_features + j] - rows_max[i]);
        }
    }

    std::vector<float> exp_sum(batch_size, 0.0);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < in_features; j++) {
            exp_sum[i] += exp_x[i * in_features + j];
        }
    }

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < in_features; j++) {
            result[i * in_features + j] = exp_x[i * in_features + j] / exp_sum[i];
        }
    }
}

void dw::reference::CPUSoftmaxBackward(const float* grad_output, const float* output, float* grad_input,
                                       size_t batch_size, size_t in_features) {
    std::vector<float> k(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < in_features; j++) {
            k[i] += grad_output[i * in_features + j] * output[i * in_features + j];
        }
    }

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < in_features; j++) {
            grad_input[i * in_features + j] = output[i * in_features + j] * (grad_output[i * in_features + j] - k[i]);
        }
    }
}

void dw::reference::CPUReLUForward(const float* in, float* out, size_t size) {
    for (size_t i = 0; i < size; i++) {
        out[i] = in[i] > 0.f ? in[i] : 0.f;
    }
}

void dw::reference::CPUReLUBackward(const float* in, const float* grad_output,
                                    float* grad_input, size_t batch_size, size_t features) {
    std::transform(in, in + batch_size * features, grad_output, grad_input,
                   [](float in, float go) { return in > 0.0 ? go : 0.0;});
}

float dw::reference::CPUCrossEntropyLossForward(const Tensor& X, const Tensor& target) {
    const auto& shape = X.shape();

    int batch_size = shape[0];
    int n_classes = shape[1];

    const float* matrix = X.data();
    const float* labels = target.data();

    float loss = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        loss -= logf(matrix[static_cast<int>(labels[i]) + (n_classes * i)]);
    }

    return loss / static_cast<float>(batch_size);
}

void dw::reference::CPUCrossEntropyLossBackward(const Tensor& X, const Tensor& target,
                                                Tensor& grad_output) {
    const auto& shape = X.shape();
    const auto& strides = X.strides();

    int batch_size = shape[0];

    const float* matrix = X.data();
    const float* labels = target.data();
    deepworks::initializer::zeros(grad_output);
    float* grad = grad_output.data();

    for (int i = 0; i < batch_size; ++i) {
        int j = static_cast<int>(labels[i] * strides[1]);
        grad[i * strides[0] + j] -= 1 / (matrix[i * strides[0] + j] * static_cast<float>(batch_size));
    }
}

void dw::reference::SGDStep(Parameters& params, float learning_rate) {
    for (auto& param: params) {
        if (param.is_trainable()) {
            float* weights = param.data().data();
            const float* grads = param.grad().data();

            const size_t size = param.data().total();

            for (size_t i = 0; i < size; ++i) {
                weights[i] -= learning_rate * grads[i];
            }
        }
    }
}

void dw::reference::CPUBatchNorm1DForward(const Tensor& input, Tensor& output,
                                          Tensor& input_centered, Tensor& std,
                                          Tensor& running_mean, Tensor& running_var,
                                          bool is_training, float eps, float alpha,
                                          const Tensor& gamma, const Tensor& beta) {
    const auto& shape = input.shape();

    int batch_size  = shape[0];
    int in_features = shape[1];

    float* raw_input          = input.data();
    float* raw_output         = output.data();
    float* raw_input_centered = input_centered.data();
    float* raw_std            = std.data();
    float* raw_running_mean   = running_mean.data();
    float* raw_running_var    = running_var.data();
    float* raw_gamma          = gamma.data();
    float* raw_beta           = beta.data();

    if (is_training) {
        std::vector<float> input_mean(in_features);
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                input_mean[j] += raw_input[j + in_features * i] / static_cast<float>(batch_size);
            }
        }

        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                raw_input_centered[j + in_features * i] = raw_input[j + in_features * i] - input_mean[j];
            }
        }

        std::vector<float> input_var(in_features);
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                input_var[j] += raw_input_centered[j + in_features * i] *
                                raw_input_centered[j + in_features * i] / static_cast<float>(batch_size);
            }
        }

        for (size_t j = 0; j < in_features; ++j) {
            raw_std[j] = std::sqrt(input_var[j] + eps);
        }

        for (size_t j = 0; j < in_features; ++j) {
            raw_running_mean[j] = raw_running_mean[j] * alpha + input_mean[j] * (1 - alpha);
            raw_running_var[j]  = raw_running_var[j] * alpha + input_var[j] * (1 - alpha);
        }
    } else {
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                raw_input_centered[j + in_features * i] = raw_input[j + in_features * i] - raw_running_mean[j];
            }
        }

        for (size_t j = 0; j < in_features; ++j) {
            raw_std[j] = std::sqrt(raw_running_var[j] + eps);
        }
    }

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            raw_output[j + in_features * i] =
                    raw_input_centered[j + in_features * i] / raw_std[j] * raw_gamma[j] + raw_beta[j];
        }
    }
}

void dw::reference::CPUBatchNorm1DBackward(const Tensor& input_centered, const Tensor& std,
                                           const Tensor& grad_output, Tensor& grad_input,
                                           const Tensor& gamma, Tensor& gamma_grad, Tensor& betta_grad) {
    const auto& shape = input_centered.shape();

    int batch_size  = shape[0];
    int in_features = shape[1];

    float* raw_input_centered = input_centered.data();
    float* raw_std            = std.data();
    float* raw_grad_output    = grad_output.data();
    float* raw_grad_input     = grad_input.data();
    float* raw_gamma          = gamma.data();
    float* raw_gamma_grad     = gamma_grad.data();
    float* raw_betta_grad     = betta_grad.data();


    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            raw_betta_grad[j] += raw_grad_output[j + in_features * i];
            raw_gamma_grad[j] += raw_input_centered[j + in_features * i] / raw_std[j] *
                                 raw_grad_output[j + in_features * i];
        }
    }

    std::vector<float> grad_x_norm(batch_size * in_features);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            grad_x_norm[j + in_features * i] = raw_grad_output[j + in_features * i] * raw_gamma[j];
        }
    }

    std::vector<float> grad_std(in_features);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            grad_std[j] -= grad_x_norm[j + in_features * i] * raw_input_centered[j + in_features * i] /
                           (raw_std[j] * raw_std[j]);
        }
    }

    std::vector<float> grad_var(in_features);
    for (size_t j = 0; j < in_features; ++j) {
        grad_var[j] = grad_std[j] / (2.0 * raw_std[j]);
    }

    std::vector<float> grad_x_centered(batch_size * in_features);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            grad_x_centered[j + in_features * i] = grad_x_norm[j + in_features * i] / raw_std[j] +
                                                   raw_input_centered[j + in_features * i] * grad_var[j] * 2.0 /
                                                   static_cast<float>(batch_size);
        }
    }

    std::vector<float> grad_mu(in_features);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            grad_mu[j] += grad_x_centered[j + in_features * i];
        }
    }

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            raw_grad_input[j + in_features * i] = grad_x_centered[j + in_features * i] - grad_mu[j] /
                                                  static_cast<float>(batch_size);
        }
    }
}

void dw::reference::MatMul(const float* in1, const float* in2, float* out, size_t m, size_t n, size_t l) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < l; j++) {
            out[i * l + j] = 0.0;
            for (size_t k = 0; k < n; k++) {
                out[i * l + j] += in1[i * n + k] * in2[k * l + j];
            }
        }
    }
}

std::vector<float> dw::reference::Transpose(const float* in, size_t rows, size_t cols) {
    std::vector<float> inT(rows * cols, 0.0);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            inT[j * rows + i] = in[i * cols + j];
        }
    }

    return std::move(inT);
}
