#include <gtest/gtest.h>

#include <deepworks/tensor.hpp>
#include <deepworks/optimizer.hpp>
#include <deepworks/parameter.hpp>

namespace dw = deepworks;

TEST(TestOptimizer, SgdStepBias) {
    const std::vector<float> weights = {5.0, 3.5, 4.2, 0.8};
    const std::vector<float> grads = {0.84, -1.23, 0.3, -0.15};
    const float learning_rate = 0.1;

    dw::Parameters m_params;

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::Tensor weight(dw::Shape{1});
        weight.data()[0] = weights[i];

        m_params.push_back(dw::Parameter(std::move(weight), true));

        dw::Tensor& grad = m_params.back().grad();
        grad.data()[0] = grads[i];
    }

    auto sgd = dw::optimizer::SGD(m_params, learning_rate);

    sgd.step();

    for (size_t i = 0; i < weights.size(); ++i) {
        float expected_weight = weights[i] - learning_rate * grads[i];
        EXPECT_FLOAT_EQ(expected_weight, m_params[i].data().data()[0]);
    }

    sgd.set_lr(sgd.get_lr() * sgd.get_lr());
    sgd.step();

    for (size_t i = 0; i < weights.size(); ++i) {
        float expected_weight = weights[i] - learning_rate * (1 + learning_rate) * grads[i];
        EXPECT_FLOAT_EQ(expected_weight, m_params[i].data().data()[0]);
    }
}

TEST(TestOptimizer, SgdStepMatrix) {
    const std::vector<std::vector<float>> weights = {
            {2.5, 1.4, 4.5, 0.15},
            {0.7, 3.2, 5.6, 2.2},
    };
    const std::vector<std::vector<float>> grads = {
            {0.84, -1.23, 0.15, 0.19},
            {0.3, -0.15, -0.2, 0.05}
    };
    const float learning_rate = 0.1;

    dw::Parameters m_params;

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::Tensor weight(dw::Shape{2, 2});
        std::copy(weights[i].begin(), weights[i].end(), weight.data());

        m_params.push_back(dw::Parameter(std::move(weight), true));

        dw::Tensor& grad = m_params.back().grad();
        std::copy(grads[i].begin(), grads[i].end(), grad.data());
    }

    auto sgd = dw::optimizer::SGD(m_params, learning_rate);

    sgd.step();

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            float expected_weight = weights[i][j] - learning_rate * grads[i][j];
            EXPECT_FLOAT_EQ(expected_weight, m_params[i].data().data()[j]);
        }
    }
}
