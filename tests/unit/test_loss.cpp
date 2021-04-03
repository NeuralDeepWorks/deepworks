#include <gtest/gtest.h>

#include <cmath>

#include "kernels_reference.hpp"
#include "test_utils.hpp"
#include <deepworks/initializers.hpp>
#include <deepworks/loss.hpp>
#include <deepworks/tensor.hpp>

namespace dw = deepworks;

TEST(TestLoss, CPUCrossEntropyLossForward) {
    const int batch_size = 2;
    const int n_classes = 3;

    const std::vector<float> labels = {0, 2};
    const std::vector<float> matrix = {
            0.23, 0.59, 0.18,
            0.09, 0.61, 0.30
    };

    dw::Tensor predictions(dw::Shape{batch_size, n_classes});
    dw::Tensor target(dw::Shape{batch_size, 1});

    std::copy(matrix.begin(), matrix.end(), predictions.data());
    std::copy(labels.begin(), labels.end(), target.data());

    auto criterion = dw::loss::CrossEntropyLoss();

    float expected_loss = dw::reference::CPUCrossEntropyLossForward(predictions, target);
    float loss = criterion.forward(predictions, target);

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

    dw::Tensor predictions(dw::Shape{batch_size, n_classes});
    dw::Tensor target(dw::Shape{batch_size, 1});
    dw::Tensor grad_output(dw::Shape{batch_size, n_classes});
    dw::Tensor reference_grad_output(dw::Shape{batch_size, n_classes});

    std::copy(matrix.begin(), matrix.end(), predictions.data());
    std::copy(labels.begin(), labels.end(), target.data());

    auto criterion = dw::loss::CrossEntropyLoss();

    dw::reference::CPUCrossEntropyLossBackward(predictions, target, reference_grad_output);
    criterion.backward(predictions, target, grad_output);

    dw::testutils::AssertTensorEqual(grad_output, reference_grad_output);
}
