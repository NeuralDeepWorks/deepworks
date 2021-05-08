#include <random>
#include <algorithm>

#include <deepworks/initializers.hpp>

#include <deepworks/utils/assert.hpp>
#include "utils/generator.hpp"

void deepworks::initializer::zeros(deepworks::Tensor& tensor) {
    std::fill_n(tensor.data(), tensor.total(), 0);
}

void deepworks::initializer::constant(deepworks::Tensor& tensor, float value) {
    std::fill_n(tensor.data(), tensor.total(), value);
}

void deepworks::initializer::xavierUniform(deepworks::Tensor& tensor) {
    auto& gen = deepworks::utils::generator();
    const auto& shape = tensor.shape();
    DeepWorks_Assert(shape.size() >= 1);
    int inp_features = shape.size() == 2 ? shape[0] : tensor.total() / shape[0];
    int out_features = shape.size() == 2 ? shape[1] : shape[0];
    auto a = std::sqrt(6.0) / (inp_features + out_features);
    std::uniform_real_distribution<float> dist(-a, a);
    std::generate_n(tensor.data(), tensor.total(), [&dist, &gen]() { return dist(gen); });
}

void deepworks::initializer::uniform(deepworks::Tensor& tensor, float lower, float upper) {
    auto& gen = deepworks::utils::generator();
    std::uniform_real_distribution<float> dist(lower, upper);
    std::generate_n(tensor.data(), tensor.total(), [&dist, &gen]() { return dist(gen); });
}
