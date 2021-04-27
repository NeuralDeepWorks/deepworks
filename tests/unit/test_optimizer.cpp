#include <gtest/gtest.h>

#include "kernels_reference.hpp"
#include "test_utils.hpp"
#include <deepworks/initializers.hpp>
#include <deepworks/tensor.hpp>
#include <deepworks/optimizer.hpp>
#include <deepworks/parameter.hpp>

namespace dw = deepworks;

struct SGDOptimizerTest : public ::testing::Test {

    void init(const std::vector<dw::Shape>& shapes) {
        for (auto&& sh : shapes) {
            params.emplace_back(dw::Tensor(sh));
            expected.emplace_back(dw::Tensor(sh));

            dw::initializer::uniform(params.back().data());
            dw::initializer::uniform(params.back().grad());
            params.back().data().copyTo(expected.back().data());
            params.back().grad().copyTo(expected.back().grad());
        }
    }

    dw::Parameters params;
    dw::Parameters expected;
};

TEST_F(SGDOptimizerTest, TestVariousShape) {
    init({dw::Shape{16, 32}, dw::Shape{10}, dw::Shape{2, 3, 4}, dw::Shape{10, 15, 16, 4}});
    float lr = 0.01;

    // Deepworks
    dw::optimizer::SGD sgd(params, lr);
    sgd.step();

    // Reference
    dw::reference::SGDStep(expected, lr);

    for (int i = 0; i < params.size(); ++i) {
        dw::testutils::AssertTensorEqual(expected[i].data(), params[i].data());
    }
}

struct MomentumTest : public ::testing::Test {

    void init(const std::vector<dw::Shape>& shapes) {
        for (auto&& sh : shapes) {
            params.emplace_back(dw::Tensor(sh));
            expected.emplace_back(dw::Tensor(sh));
            velocities.emplace_back(dw::Tensor(sh));

            dw::initializer::uniform(params.back().data());
            dw::initializer::uniform(params.back().grad());
            dw::initializer::zeros(velocities.back());
            params.back().data().copyTo(expected.back().data());
            params.back().grad().copyTo(expected.back().grad());
        }
    }

    dw::Parameters          params;
    dw::Parameters          expected;
    std::vector<dw::Tensor> velocities;
};

TEST_F(MomentumTest, TestVariousShape) {
    init({dw::Shape{4, 16}, dw::Shape{32}, dw::Shape{4, 5, 6}, dw::Shape{32, 8, 28, 28}});
    float lr    = 0.01;
    float gamma = 0.9;

    // Deepworks
    dw::optimizer::SGDMomentum opt(params, lr, gamma);
    opt.step(); // the first step take zeros velocities
    opt.step();

    // Reference
    dw::reference::SGDMomentumStep(expected, velocities, lr, gamma);
    dw::reference::SGDMomentumStep(expected, velocities, lr, gamma);

    for (int i = 0; i < params.size(); ++i) {
        dw::testutils::AssertTensorEqual(expected[i].data(), params[i].data());
    }
}

struct AdamTest : public ::testing::Test {

    void init(const std::vector<dw::Shape>& shapes) {
        for (auto&& sh : shapes) {
            params.emplace_back(dw::Tensor(sh));
            expected.emplace_back(dw::Tensor(sh));
            moving_mean.emplace_back(dw::Tensor(sh));
            moving_variance.emplace_back(dw::Tensor(sh));

            dw::initializer::uniform(params.back().data());
            dw::initializer::uniform(params.back().grad());
            dw::initializer::zeros(moving_mean.back());
            dw::initializer::zeros(moving_variance.back());
            params.back().data().copyTo(expected.back().data());
            params.back().grad().copyTo(expected.back().grad());
        }
    }

    dw::Parameters          params;
    dw::Parameters          expected;
    std::vector<dw::Tensor> moving_mean;
    std::vector<dw::Tensor> moving_variance;
};

TEST_F(AdamTest, TestVariousShape) {
    init({dw::Shape{4, 16}, dw::Shape{32}, dw::Shape{4, 5, 6}, dw::Shape{32, 8, 28, 28}});
    float lr          = 1e-2;
    std::array<float, 2> betas = {0.9f, 0.999f};
    float epsilon     = 0.999;

    // Deepworks
    dw::optimizer::Adam opt(params, lr, betas, epsilon);
    opt.step(); // the first step take zeros moving_mean && moving_variance
    opt.step();

    // Reference
    dw::reference::AdamStep(expected, moving_mean, moving_variance, lr, betas, epsilon, 1u);
    dw::reference::AdamStep(expected, moving_mean, moving_variance, lr, betas, epsilon, 2u);

    for (int i = 0; i < params.size(); ++i) {
        dw::testutils::AssertTensorEqual(expected[i].data(), params[i].data());
    }
}
