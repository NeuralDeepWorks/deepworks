#include <gtest/gtest.h>

#include <cmath>

#include "kernels_reference.hpp"
#include "test_utils.hpp"
#include <deepworks/tensor.hpp>
#include <deepworks/loss.hpp>

namespace dw = deepworks;

TEST(TestLoss, CPUCrossEntropyLossForward) {
    const int batch_size = 2;
    const int n_classes = 3;

    const std::vector<float> labels = {0, 2};
    const std::vector<float> matrix = {
            0.23, 0.59, 0.18,
            0.09, 0.61, 0.30
    };

    dw::Tensor X(dw::Shape{batch_size, n_classes});
    dw::Tensor target(dw::Shape{batch_size});

    std::copy(matrix.begin(), matrix.end(), X.data());
    std::copy(labels.begin(), labels.end(), target.data());

    float expected_loss = dw::reference::CPUCrossEntropyLossForward(X, target);
    float loss = dw::loss::CPUCrossEntropyLossForward(X, target);

    EXPECT_FLOAT_EQ(loss, expected_loss);
}

TEST(TestLoss, CPUCrossEntropyLossBackward) {
    const int batch_size = 4;
    const int n_classes = 2;

    const std::vector<float> labels = {0, 1, 1, 0};
    const std::vector<float> matrix = {
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

    dw::testutils::AssertTensorEqual(grad_output, reference_grad_output);
}
