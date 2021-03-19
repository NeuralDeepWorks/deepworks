#include <gtest/gtest.h>

#include <cmath>

#include "kernels_reference.hpp"
#include <deepworks/tensor.hpp>
#include <deepworks/loss.hpp>

namespace dw = deepworks;

TEST(TestLoss, CPUCrossEntropyLossForward) {
    int batch_size = 2;
    int n_classes = 3;

    std::vector<float> labels = {0, 2};
    std::vector<float> matrix = {
            0.23, 0.59, 0.18,
            0.09, 0.61, 0.30
    };

    dw::Tensor X(dw::Shape{batch_size, n_classes});
    dw::Tensor target(dw::Shape{batch_size});

    std::copy(matrix.begin(), matrix.end(), X.data());
    std::copy(labels.begin(), labels.end(), target.data());

    float expected_loss = dw::reference::CPUCrossEntropyLossForward(X, target);
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

    dw::Tensor X(dw::Shape{batch_size, n_classes});
    dw::Tensor target(dw::Shape{batch_size});
    dw::Tensor grad_output(dw::Shape{batch_size, n_classes});
    dw::Tensor reference_grad_output(dw::Shape{batch_size, n_classes});

    std::copy(matrix.begin(), matrix.end(), X.data());
    std::copy(labels.begin(), labels.end(), target.data());
    std::fill_n(grad_output.data(), batch_size * n_classes, 0);
    std::fill_n(reference_grad_output.data(), batch_size * n_classes, 0);

    dw::reference::CPUCrossEntropyLossBackward(X, target, reference_grad_output);
    dw::loss::CPUCrossEntropyLossBackward(X, target, grad_output);

    float* grad_output_data = grad_output.data();
    float* reference_grad_output_data = reference_grad_output.data();

    for (int i = 0; i < batch_size * n_classes; ++i) {
        EXPECT_FLOAT_EQ(reference_grad_output_data[i], grad_output_data[i]);
    }
}
