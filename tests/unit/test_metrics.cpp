#include <gtest/gtest.h>

#include <deepworks/tensor.hpp>
#include <deepworks/metrics.hpp>

namespace dw = deepworks;

TEST(TestMetrics, accuracy) {
    std::vector<float> labels = {0, 1, 0, 0};
    std::vector<float> predict = {
        0.56, 0.44,
        0.03, 0.97,
        0.57, 0.43,
        0.18, 0.82,
    };

    dw::Tensor y_pred(dw::Shape{4, 2});
    dw::Tensor y_true(dw::Shape{4});

    std::copy(labels.begin(), labels.end(), y_true.data());
    std::copy(predict.begin(), predict.end(), y_pred.data());

    float expected = 0.75;
    float acc = dw::metric::accuracy(y_pred, y_true);
    EXPECT_FLOAT_EQ(expected, acc);
}

TEST(TestMetrics, sparse_accuracy) {
    std::vector<float> labels = {
        1, 0,
        0, 1,
        0, 1,
        1, 0,
    };
    std::vector<float> predict = {
        0.56, 0.44,
        0.03, 0.97,
        0.57, 0.43,
        0.18, 0.82,
    };

    dw::Tensor y_pred(dw::Shape{4, 2});
    dw::Tensor y_true(dw::Shape{4, 2});

    std::copy(labels.begin(), labels.end(), y_true.data());
    std::copy(predict.begin(), predict.end(), y_pred.data());

    float expected = 0.5;
    float acc = dw::metric::sparse_accuracy(y_pred, y_true);
    EXPECT_FLOAT_EQ(expected, acc);
}
