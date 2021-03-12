#include <gtest/gtest.h>
#include <random>
#include <deepworks/deepworks.hpp>
#include "kernels_reference.hpp"

namespace dw = deepworks;

// FIXME: It's initial tests implementation,
// will be more generic and elegant in future.
namespace {
dw::Tensor fillRandom(dw::Tensor& tensor) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1, 1);

    float* p = tensor.data();
    for (int i = 0; i < tensor.total(); ++i) {
        p[i] = dist(gen);
    }

    return tensor;
}

void expect_eq(const dw::Tensor& actual, const dw::Tensor& expected) {
    ASSERT_EQ(actual.shape()  , expected.shape());
    ASSERT_EQ(actual.strides(), expected.strides());

    auto* actual_p   = actual.data();
    auto* expected_p = expected.data();

    auto total = actual.total();
    for (int i = 0; i < total; ++i) {
        EXPECT_EQ(expected_p[i], actual_p[i]);
    }
}

} // anonymous namespace 

TEST(LayerTests, CPUReLU) {
    int n_features = 100;
    dw::Placeholder in(dw::Shape{1, n_features});
    dw::Model model(in, dw::ReLU("relu")(in));

    dw::Tensor input(in.shape());
    fillRandom(input);

    model.compile();

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    deepworks::reference::CPUReLUForward(input.data(), expected.data(), input.total());

    expect_eq(actual, expected);
}
