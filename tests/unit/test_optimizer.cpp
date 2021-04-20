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
