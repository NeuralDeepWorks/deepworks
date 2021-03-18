#include <gtest/gtest.h>

#include <cmath>

#include <deepworks/tensor.hpp>
#include <deepworks/loss.hpp>

namespace dw = deepworks;

TEST(TestLoss, CPUCrossEntropyLossForward) {
    int cols = 3;
    int rows = 2;
    std::vector<float> labels = {0, 2};
    std::vector<float> matrix = {
            0.23, 0.59, 0.18,
            0.09, 0.61, 0.30
    };

    dw::Tensor X(dw::Shape{rows, cols});
    dw::Tensor target(dw::Shape{rows});

    std::copy(matrix.begin(), matrix.end(), X.data());
    std::copy(labels.begin(), labels.end(), target.data());

    float expected_loss = 0;
    for (size_t i = 0; i < rows; ++i) {
        expected_loss -= logf(matrix[labels[i] + static_cast<float>(cols * i)]);
    }
    expected_loss /= static_cast<float>(rows);

    float loss = dw::losses::CPUCrossEntropyLossForward(X, target);

    EXPECT_FLOAT_EQ(expected_loss, loss);
}

TEST(TestLoss, CPUCrossEntropyLossBackward) {
    int cols = 2;
    int rows = 4;
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

    dw::Tensor X(dw::Shape{rows, cols});
    dw::Tensor target(dw::Shape{rows});
    dw::Tensor grad_output(dw::Shape{rows, cols});

    std::copy(matrix.begin(), matrix.end(), X.data());
    std::copy(labels.begin(), labels.end(), target.data());
    std::copy(gradient.begin(), gradient.end(), grad_output.data());

    gradient = matrix;
    for (int i = 0; i < rows; ++i) {
        gradient[i * cols + static_cast<int>(labels[i])] -= 1;
    }
    for (int i = 0; i < rows * cols; ++i) {
        gradient[i] /= static_cast<float>(rows);
    }

    float* grad_output_data = grad_output.data();

    dw::losses::CPUCrossEntropyLossBackward(X, target, grad_output);

    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_FLOAT_EQ(gradient[i], grad_output_data[i]);
    }
}
