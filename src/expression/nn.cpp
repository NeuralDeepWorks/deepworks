#include <numeric>

#include <deepworks/nn.hpp>
#include <deepworks/call.hpp>
#include <deepworks/tensor.hpp>
#include <deepworks/initializers.hpp>

#include "util/assert.hpp"

enum Input  {N, C, H, W};
enum Kernel {KH, KW};

deepworks::Linear::Linear(int units, std::string name)
    : BaseOp<deepworks::Linear>(deepworks::LayerInfo(std::move(name), "Linear")) {
    m_info.impl().attrs["units"] = units;
}

void deepworks::Linear::init(const Shape& in_shape) {
    int units = m_info.impl().attrs["units"].get<int>();

    auto second_shape = std::accumulate(in_shape.begin() + 1, in_shape.end(), 1, std::multiplies<int>());
    // NB: Init weight.
    deepworks::Tensor weight(deepworks::Shape{units, second_shape});
    deepworks::initializer::xavierUniform(weight);
    m_info.impl().params.emplace_back(std::move(weight), true);

    // NB: Init bias.
    deepworks::Tensor bias(deepworks::Shape{units});
    deepworks::initializer::zeros(bias);
    m_info.impl().params.emplace_back(std::move(bias), true);
}

deepworks::Shape deepworks::Linear::outShape(const deepworks::Shape& in_shape) {
    DeepWorks_Assert(in_shape.size() != 1u && "Linear layer doesn't work with 1D tensors");
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

deepworks::MaxPooling::MaxPooling(const std::array<int, 2>& kernel,
                                  const std::array<int, 2>& padding,
                                  const std::array<int, 2>& stride,
                                  std::string name)
    : BaseOp<deepworks::MaxPooling>(LayerInfo(std::move(name), "MaxPooling")) {
    m_info.impl().attrs["kernel"] = kernel;
    m_info.impl().attrs["padding"] = padding;
    m_info.impl().attrs["stride"] = stride;
}

deepworks::Shape deepworks::MaxPooling::outShape(const deepworks::Shape& in_shape) {
    DeepWorks_Assert(in_shape.size() == 4u && "MaxPooling layer works only with 4D tensors");
    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    int h_out = (in_shape[Input::H] + 2 * padding[Kernel::KH] - kernel[Kernel::KH]) / stride[Kernel::KH] + 1;
    int w_out = (in_shape[Input::W] + 2 * padding[Kernel::KW] - kernel[Kernel::KW]) / stride[Kernel::KW] + 1;
    return {in_shape[0], in_shape[1], h_out, w_out};
}

deepworks::Convolution::Convolution(int out_channels,
                                    const std::array<int, 2>& kernel,
                                    const std::array<int, 2>& padding,
                                    const std::array<int, 2>& stride,
                                    std::string name)
    : BaseOp<deepworks::Convolution>(LayerInfo(std::move(name), "Convolution")) {
    m_info.impl().attrs["out_channels"] = out_channels;
    m_info.impl().attrs["kernel"] = kernel;
    m_info.impl().attrs["padding"] = padding;
    m_info.impl().attrs["stride"] = stride;
}

void deepworks::Convolution::init(const Shape& in_shape) {
    int out_channels = m_info.impl().attrs["out_channels"].get<int>();
    auto kernel = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();

    // NB: Init weight.
    deepworks::Tensor weight({out_channels, in_shape[Input::C], kernel[Kernel::KH], kernel[Kernel::KW]});
    deepworks::initializer::xavierUniform(weight);
    m_info.impl().params.emplace_back(std::move(weight), true);

    // NB: Init bias.
    deepworks::Tensor bias({out_channels});
    deepworks::initializer::zeros(bias);
    m_info.impl().params.emplace_back(std::move(bias), true);
}

deepworks::Shape deepworks::Convolution::outShape(const deepworks::Shape& in_shape) {
    DeepWorks_Assert(in_shape.size() == 4u && "Convolution layer works only with 4D tensors");
    int out_channels = m_info.impl().attrs["out_channels"].get<int>();
    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    int h_out = (in_shape[Input::H] + 2 * padding[Kernel::KH] - kernel[Kernel::KH]) / stride[Kernel::KH] + 1;
    int w_out = (in_shape[Input::W] + 2 * padding[Kernel::KW] - kernel[Kernel::KW]) / stride[Kernel::KW] + 1;
    return {in_shape[0], out_channels, h_out, w_out};
}
