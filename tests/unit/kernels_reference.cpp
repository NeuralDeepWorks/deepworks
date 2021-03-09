#include "kernels_reference.hpp"

#include <limits>
#include <cmath>

void deepworks::reference::CPULinearForward(const float* X, const float* W, float* result,
                                            size_t batch_size, size_t in_features, size_t out_features) {

    deepworks::reference::Multiply(X, W, result, batch_size, in_features, out_features);
}

void deepworks::reference::CPULinearAddBias(const float* b, float* result, size_t batch_size, size_t out_features) {

    for (size_t sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        for (size_t i = 0; i < out_features; i++) {
            result[sample_idx * out_features + i] += b[i];
        }
    }
}

void deepworks::reference::CPULinearBackward(const float* input, const float* W, const float* dx, float* dW, float* grad_output,
                                             size_t batch_size, size_t in_features, size_t out_features) {

    auto inputT = deepworks::reference::Transpose(input, batch_size, in_features);

    deepworks::reference::Multiply(inputT.data(), dx, dW, in_features, batch_size, out_features);
    for (size_t i = 0; i < in_features * out_features; i++) {
        dW[i] /= batch_size;
    }

    auto WT = deepworks::reference::Transpose(W, in_features, out_features);

    deepworks::reference::Multiply(dx, WT.data(), grad_output, batch_size, out_features, in_features);
}

void deepworks::reference::CPULinearBiasBackward(const float* dx, float* db, size_t batch_size, size_t out_features) {

    for (size_t j = 0; j < out_features; j++) {
        float sum = 0.0;
        for (size_t i = 0; i < batch_size; i++) {
            sum += dx[i * batch_size + j];
        }
        db[j] = sum / batch_size;
    }
}

void deepworks::reference::CPUSoftmaxForward(const float* X, float* result, size_t batch_size, size_t in_features) {

    std::vector<float> rows_max(batch_size, std::numeric_limits<float>::min());

    // find max feature for each sample
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

void deepworks::reference::CPUSoftmaxBackward(const float* dx, const float* output, float* grad_output,
                                              size_t batch_size, size_t in_features) {
    std::vector<float> k(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < in_features; j++) {
            k[i] += dx[i * in_features + j] * output[i * in_features + j];
        }
    }

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < in_features; j++) {
            grad_output[i * in_features + j] = output[i * in_features + j] * (dx[i * in_features + j] - k[i]);
        }
    }
}

void deepworks::reference::CPUReLUForward(const float* in, float* out, size_t size) {

    for (size_t i = 0; i < size; i++) {
        out[i] = in[i] > 0.0 ? in[i] : 0.0;
    }
}

void deepworks::reference::CPUReLUBackward(const float* dx, const float* output,
                                           float* grad_output, size_t size) {

    for (size_t i = 0; i < size; i++) {
        if (output[i] > 0.0) {
            grad_output[i] = dx[i];
        } else {
            grad_output[i] = 0.0;
        }
    }
}

void deepworks::reference::Multiply(const float* in1, const float* in2, float* out, size_t m, size_t n, size_t l) {

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < l; j++) {
            out[i * l + j] = 0.0;
            for (size_t k = 0; k < n; k++) {
                out[i * l + j] += in1[i * n + k] * in2[k * l + j];
            }
        }
    }
}

std::vector<float> deepworks::reference::Transpose(const float* in, size_t rows, size_t cols) {

    std::vector<float> inT(rows * cols, 0.0);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            inT[j * rows + i] = in[i * cols + j];
        }
    }

    return std::move(inT);
}
