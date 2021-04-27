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
        int idx = 0;
        for (auto&& sh : shapes) {
            std::string name = "param" + std::to_string(idx++);
            auto param = params.emplace  (name, dw::Tensor(sh)).first->second;
            auto ex    = expected.emplace(name, dw::Tensor(sh)).first->second;

            dw::initializer::uniform(param.data());
            dw::initializer::uniform(param.grad());

            param.data().copyTo(ex.data());
            param.grad().copyTo(ex.grad());
        }
    }

    dw::ParamMap params;
    dw::ParamMap expected;
};

TEST_F(SGDOptimizerTest, TestVariousShape) {
    init({dw::Shape{16, 32}, dw::Shape{10}, dw::Shape{2, 3, 4}, dw::Shape{10, 15, 16, 4}});
    float lr = 0.01;

    // Deepworks
    dw::optimizer::SGD sgd(params, lr);
    sgd.step();

    // Reference
    dw::reference::SGDStep(expected, lr);

    for (auto& [name, ex] : expected) {
        dw::testutils::AssertTensorEqual(ex.data(), params.at(name).data());
    }
}

struct MomentumTest : public ::testing::Test {

    void init(const std::vector<dw::Shape>& shapes) {
        int idx = 0;
        for (auto&& sh : shapes) {
            std::string name = "param" + std::to_string(idx++);
            auto param = params.emplace  (name, dw::Tensor(sh)).first->second;
            auto ex    = expected.emplace(name, dw::Tensor(sh)).first->second;

            velocities.emplace(name, dw::Tensor::zeros(sh));

            dw::initializer::uniform(param.data());
            dw::initializer::uniform(param.grad());

            param.data().copyTo(ex.data());
            param.grad().copyTo(ex.grad());
        }
    }

    dw::ParamMap   params;
    dw::ParamMap   expected;
    dw::TensorMap  velocities;
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

    for (auto& [name, ex] : expected) {
        dw::testutils::AssertTensorEqual(ex.data(), params.at(name).data());
    }
}

struct AdamTest : public ::testing::Test {

    void init(const std::vector<dw::Shape>& shapes) {
        int idx = 0;
        for (auto&& sh : shapes) {
            std::string name = "param" + std::to_string(idx++);
            auto param = params.emplace  (name, dw::Tensor(sh)).first->second;
            auto ex    = expected.emplace(name, dw::Tensor(sh)).first->second;

            moving_mean.emplace    (name, dw::Tensor::zeros(sh));
            moving_variance.emplace(name, dw::Tensor::zeros(sh));

            dw::initializer::uniform(param.data());
            dw::initializer::uniform(param.grad());

            param.data().copyTo(ex.data());
            param.grad().copyTo(ex.grad());
        }
    }

    dw::ParamMap  params;
    dw::ParamMap  expected;
    dw::TensorMap moving_mean;
    dw::TensorMap moving_variance;
};

TEST_F(AdamTest, TestVariousShape) {
    init({dw::Shape{4, 16}, dw::Shape{32}, dw::Shape{4, 5, 6}, dw::Shape{32, 8, 28, 28}});

    float lr                   = 1e-2;
    float epsilon              = 0.999;
    std::array<float, 2> betas = {0.9f, 0.999f};

    // Deepworks
    dw::optimizer::Adam opt(params, lr, betas, epsilon);
    opt.step(); // the first step take zeros moving_mean && moving_variance
    opt.step();

    // Reference
    dw::reference::AdamStep(expected, moving_mean, moving_variance, lr, betas, epsilon, 1u);
    dw::reference::AdamStep(expected, moving_mean, moving_variance, lr, betas, epsilon, 2u);

    for (auto[name, ex] : expected) {
        dw::testutils::AssertTensorEqual(ex.data(), params.at(name).data());
    }
}
