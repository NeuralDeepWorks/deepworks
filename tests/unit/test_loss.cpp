#include <gtest/gtest.h>

#include <cmath>

#include <deepworks/tensor.hpp>
#include <deepworks/loss.hpp>

namespace dw = deepworks;

TEST(TestLoss, CPUCrossEntropyLossForward) {
    int n_classes = 3;
    int batch_size = 2;
    std::vector<float> labels = {0, 2};
    std::vector<float> matrix = {
            0.23, 0.59, 0.18,
            0.09, 0.61, 0.30
    };

    dw::Tensor X(dw::Shape{batch_size, n_classes});
    dw::Tensor target(dw::Shape{batch_size});

    std::copy(matrix.begin(), matrix.end(), X.data());
    std::copy(labels.begin(), labels.end(), target.data());

    float expected_loss = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        expected_loss -= logf(matrix[labels[i] + static_cast<float>(n_classes * i)]);
    }
    expected_loss /= static_cast<float>(batch_size);

    float loss = dw::loss::CPUCrossEntropyLossForward(X, target);

    EXPECT_FLOAT_EQ(expected_loss, loss);
}

TEST(TestLoss, CPUCrossEntropyLossBackward) {
    int batch_size = 4;
    int n_classes = 2;

    std::vector<float> labels = {0, 1, 1, 0};
    std::vector<float> matrix = {
            0.61, 0.39,
            0.18, 0.82,
            0.51, 0.49,
            0.05, 0.95
    };
    std::vector<float> gradient = {
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0
    };

    dw::Tensor X(dw::Shape{batch_size, n_classes});
    dw::Tensor target(dw::Shape{batch_size});
    dw::Tensor grad_output(dw::Shape{batch_size, n_classes});

    std::copy(matrix.begin(), matrix.end(), X.data());
    std::copy(labels.begin(), labels.end(), target.data());
    std::copy(gradient.begin(), gradient.end(), grad_output.data());

    for (int i = 0; i < batch_size; ++i) {
        int index = i * n_classes + static_cast<int>(labels[i]);
        gradient[index] -= 1 / (matrix[index] * static_cast<float>(batch_size));
    }


    dw::loss::CPUCrossEntropyLossBackward(X, target, grad_output);
    float* grad_output_data = grad_output.data();

    for (int i = 0; i < batch_size * n_classes; ++i) {
        EXPECT_FLOAT_EQ(gradient[i], grad_output_data[i]);
    }
}
