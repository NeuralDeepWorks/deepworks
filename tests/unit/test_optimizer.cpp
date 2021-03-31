#include <gtest/gtest.h>

#include "kernels_reference.hpp"
#include "test_utils.hpp"
#include <deepworks/tensor.hpp>
#include <deepworks/optimizer.hpp>
#include <deepworks/parameter.hpp>

namespace dw = deepworks;

TEST(TestOptimizer, SgdStepBias) {
    const std::vector<float> weights = {5.0, 3.5, 4.2, 0.8};
    const std::vector<float> grads = {0.84, -1.23, 0.3, -0.15};
    const float learning_rate = 0.1;

    dw::Parameters params;
    dw::Parameters reference_params;

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::Tensor weight(dw::Shape{1});
        dw::Tensor reference_weight(dw::Shape{1});
        *weight.data() = weights[i];
        *reference_weight.data() = weights[i];

        params.emplace_back(dw::Parameter(std::move(weight), true));
        reference_params.emplace_back(dw::Parameter(std::move(reference_weight), true));

        dw::Tensor& grad = params.back().grad();
        dw::Tensor& reference_grad = reference_params.back().grad();
        *grad.data() = grads[i];
        *reference_grad.data() = grads[i];
    }

    auto sgd = dw::optimizer::SGD(params, learning_rate);
    sgd.step();

    dw::reference::SGDStep(reference_params, learning_rate);

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::testutils::AssertTensorEqual(reference_params[i].data(), params[i].data());
    }

    sgd.set_lr(sgd.get_lr() * sgd.get_lr());
    sgd.step();

    dw::reference::SGDStep(reference_params, learning_rate * learning_rate);

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::testutils::AssertTensorEqual(reference_params[i].data(), params[i].data());
    }
}

TEST(TestOptimizer, SgdEvenNotTrainable) {
    const std::vector<float> weights = {5.0, 3.5, 4.2, 0.8};
    const std::vector<float> grads = {0.84, -1.23, 0.3, -0.15};
    const float learning_rate = 0.1;

    dw::Parameters params;
    dw::Parameters reference_params;
    bool trainable = false;

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::Tensor weight(dw::Shape{1});
        dw::Tensor reference_weight(dw::Shape{1});
        *weight.data() = weights[i];
        *reference_weight.data() = weights[i];

        params.emplace_back(dw::Parameter(std::move(weight), trainable));
        reference_params.emplace_back(dw::Parameter(std::move(reference_weight), trainable));

        dw::Tensor& grad = params.back().grad();
        dw::Tensor& reference_grad = reference_params.back().grad();
        *grad.data() = grads[i];
        *reference_grad.data() = grads[i];

        trainable = !trainable;
    }

    auto sgd = dw::optimizer::SGD(params, learning_rate);
    sgd.step();

    dw::reference::SGDStep(reference_params, learning_rate);

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::testutils::AssertTensorEqual(reference_params[i].data(), params[i].data());
    }
}

TEST(TestOptimizer, SgdStepMatrix2d) {
    const std::vector<std::vector<float>> weights = {
            {2.5, 1.4, 4.5, 0.15},
            {0.7, 3.2, 5.6, 2.2},
    };
    const std::vector<std::vector<float>> grads = {
            {0.84, -1.23, 0.15, 0.19},
            {0.3,  -0.15, -0.2, 0.05}
    };
    const float learning_rate = 0.1;

    dw::Parameters params;
    dw::Parameters reference_params;

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::Tensor weight(dw::Shape{2, 2});
        dw::Tensor reference_weight(dw::Shape{2, 2});
        std::copy(weights[i].begin(), weights[i].end(), weight.data());
        std::copy(weights[i].begin(), weights[i].end(), reference_weight.data());

        params.emplace_back(dw::Parameter(std::move(weight), true));
        reference_params.emplace_back(dw::Parameter(std::move(reference_weight), true));

        dw::Tensor& grad = params.back().grad();
        dw::Tensor& reference_grad = reference_params.back().grad();
        std::copy(grads[i].begin(), grads[i].end(), grad.data());
        std::copy(grads[i].begin(), grads[i].end(), reference_grad.data());
    }

    auto sgd = dw::optimizer::SGD(params, learning_rate);
    sgd.step();

    dw::reference::SGDStep(reference_params, learning_rate);

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::testutils::AssertTensorEqual(reference_params[i].data(), params[i].data());
    }
}

TEST(TestOptimizer, SgdStepMatrix4d) {
    // 2 * 2 * 1 * 3 = 12
    const std::vector<std::vector<float>> weights = {
            {-0.32, -0.15, 0.44, 0.30, 0.51, -5.92, 4.31, 0.26, 0.03,  1.58, 2.61,  0.34},
            {1.61,  0.00,  0.14, 0.19, 0.99, -2.01, 2.11, 1.58, -0.75, 2.11, -0.53, 1.71}
    };
    const std::vector<std::vector<float>> grads = {
            {0.04, -0.5, 0.61,  0.15, 0.03, -2.02, 0.17,  0.66, 0.50, -1.8, 0.00, 0.01},
            {1.05, 0.33, -0.54, 0.32, 0.14, 0.50,  -1.07, 0.43, 0.15, 3.08, 0.07, 1.09}
    };
    const float learning_rate = 0.01;

    dw::Parameters params;
    dw::Parameters reference_params;

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::Tensor weight(dw::Shape{2, 2, 1, 3});
        dw::Tensor reference_weight(dw::Shape{2, 2, 1, 3});
        std::copy(weights[i].begin(), weights[i].end(), weight.data());
        std::copy(weights[i].begin(), weights[i].end(), reference_weight.data());

        params.emplace_back(dw::Parameter(std::move(weight), true));
        reference_params.emplace_back(dw::Parameter(std::move(reference_weight), true));

        dw::Tensor& grad = params.back().grad();
        dw::Tensor& reference_grad = reference_params.back().grad();
        std::copy(grads[i].begin(), grads[i].end(), grad.data());
        std::copy(grads[i].begin(), grads[i].end(), reference_grad.data());
    }

    auto sgd = dw::optimizer::SGD(params, learning_rate);
    sgd.step();

    dw::reference::SGDStep(reference_params, learning_rate);

    for (size_t i = 0; i < weights.size(); ++i) {
        dw::testutils::AssertTensorEqual(reference_params[i].data(), params[i].data());
    }
}
