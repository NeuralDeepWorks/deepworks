#include <deepworks/nn.hpp>
#include <deepworks/call.hpp>
#include <deepworks/tensor.hpp>
#include <deepworks/initializers.hpp>

#include "util/assert.hpp"

deepworks::Linear::Linear(int units, std::string name)
    : BaseOp<deepworks::Linear>(deepworks::LayerInfo(std::move(name), "Linear")) {
    m_info.impl().attrs["units"] = units;
}

void deepworks::Linear::init(const Shape& in_shape) {
    int units = m_info.impl().attrs["units"].get<int>();

    // NB: Init weight.
    deepworks::Tensor weight(deepworks::Shape{units, in_shape[1]});
    deepworks::initializer::xavierUniform(weight);
    m_info.impl().params.emplace_back(std::move(weight), true);

    // NB: Init bias.
    deepworks::Tensor bias(deepworks::Shape{units});
    deepworks::initializer::zeros(bias);
    m_info.impl().params.emplace_back(std::move(bias), true);
}

deepworks::Shape deepworks::Linear::outShape(const deepworks::Shape& in_shape) {
    DeepWorks_Assert(in_shape.size() == 2u && "Linear layer works only with 2D tensors");
    int units = m_info.impl().attrs["units"].get<int>();
    return {in_shape[0], units};
}

deepworks::ReLU::ReLU(std::string name)
    : BaseOp<deepworks::ReLU>(deepworks::LayerInfo(std::move(name), "ReLU")) {
}

deepworks::Shape deepworks::ReLU::outShape(const deepworks::Shape& in_shape) {
    return in_shape;
}

deepworks::Softmax::Softmax(std::string name)
    : BaseOp<deepworks::Softmax>(LayerInfo(std::move(name), "Softmax")) {
}

deepworks::Shape deepworks::Softmax::outShape(const deepworks::Shape& in_shape) {
    return in_shape;
}

deepworks::BatchNorm1D::BatchNorm1D(float eps, float alpha, std::string name)
    : BaseOp<deepworks::BatchNorm1D>(LayerInfo(std::move(name), "BatchNorm1D")) {
    m_info.impl().attrs["eps"] = eps;
    m_info.impl().attrs["alpha"] = alpha;
}

void deepworks::BatchNorm1D::init(const Shape& in_shape) {
    // NB: Init gamma
    deepworks::Tensor gamma(deepworks::Shape{in_shape[1]});
    deepworks::initializer::constant(gamma, 1.0);
    m_info.impl().params.emplace_back(std::move(gamma), true);

    // NB: Init beta.
    deepworks::Tensor beta(deepworks::Shape{in_shape[1]});
    deepworks::initializer::zeros(beta);
    m_info.impl().params.emplace_back(std::move(beta), true);
}

deepworks::Shape deepworks::BatchNorm1D::outShape(const deepworks::Shape& in_shape) {
    return in_shape;
}

deepworks::ELU::ELU(float alpha, std::string name)
    : BaseOp<deepworks::ELU>(deepworks::LayerInfo(std::move(name), "ELU")) {
    m_info.impl().attrs["alpha"] = alpha;
}

deepworks::Shape deepworks::ELU::outShape(const deepworks::Shape& in_shape) {
    return in_shape;
}
