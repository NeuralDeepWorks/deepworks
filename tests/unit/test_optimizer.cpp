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

            velocities.emplace_back(dw::Tensor::zeros(sh));

            dw::initializer::uniform(param.data());
            dw::initializer::uniform(param.grad());

            param.data().copyTo(ex.data());
            param.grad().copyTo(ex.grad());
        }
    }

    dw::ParamMap            params;
    dw::ParamMap            expected;
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

    for (auto& [name, ex] : expected) {
        dw::testutils::AssertTensorEqual(ex.data(), params.at(name).data());
    }
}
