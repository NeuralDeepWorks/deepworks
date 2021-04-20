#include <deepworks/nn.hpp>
#include <deepworks/call.hpp>
#include <deepworks/tensor.hpp>
#include <deepworks/initializers.hpp>

#include "util/assert.hpp"

enum Input  {N, C, H, W};
enum Kernel {KH, KW};

namespace dw = deepworks;

dw::Linear::Linear(int units, std::string name)
    : BaseOp<dw::Linear>(dw::LayerInfo(std::move(name), "Linear")) {
    m_info.impl().attrs["units"] = units;
}

void dw::Linear::init(const Shape& in_shape) {
    int units = m_info.impl().attrs["units"].get<int>();
    // NB: Init weight.
    m_info.impl().params.emplace("weight", dw::Tensor::xavierUniform({units, in_shape[1]}));
    // NB: Init bias.
    m_info.impl().params.emplace("bias", dw::Tensor::zeros({units}));
}

dw::Shape dw::Linear::outShape(const dw::Shape& in_shape) {
    DeepWorks_Assert(in_shape.size() == 2u && "Linear layer works only with 2D tensors");
    int units = m_info.impl().attrs["units"].get<int>();
    return {in_shape[0], units};
}

dw::ReLU::ReLU(std::string name)
    : BaseOp<dw::ReLU>(dw::LayerInfo(std::move(name), "ReLU")) {
}

dw::Shape dw::ReLU::outShape(const dw::Shape& in_shape) {
    return in_shape;
}

dw::Softmax::Softmax(std::string name)
    : BaseOp<dw::Softmax>(LayerInfo(std::move(name), "Softmax")) {
}

dw::Shape dw::Softmax::outShape(const dw::Shape& in_shape) {
    return in_shape;
}

dw::BatchNorm1D::BatchNorm1D(float eps, float alpha, std::string name)
    : BaseOp<dw::BatchNorm1D>(LayerInfo(std::move(name), "BatchNorm1D")) {
    m_info.impl().attrs["eps"] = eps;
    m_info.impl().attrs["alpha"] = alpha;
}

void dw::BatchNorm1D::init(const Shape& in_shape) {
    // NB: Init gamma
    m_info.impl().params.emplace("gamma", dw::Tensor::constant({in_shape[1]}, 1.0));
    // NB: Init beta.
    m_info.impl().params.emplace("beta", dw::Tensor::zeros({in_shape[1]}));
}

dw::Shape dw::BatchNorm1D::outShape(const dw::Shape& in_shape) {
    return in_shape;
}

dw::ELU::ELU(float alpha, std::string name)
    : BaseOp<dw::ELU>(dw::LayerInfo(std::move(name), "ELU")) {
    m_info.impl().attrs["alpha"] = alpha;
}

dw::Shape dw::ELU::outShape(const dw::Shape& in_shape) {
    return in_shape;
}

dw::MaxPooling::MaxPooling(const std::array<int, 2>& kernel,
                                  const std::array<int, 2>& padding,
                                  const std::array<int, 2>& stride,
                                  std::string name)
    : BaseOp<dw::MaxPooling>(LayerInfo(std::move(name), "MaxPooling")) {
    m_info.impl().attrs["kernel"] = kernel;
    m_info.impl().attrs["padding"] = padding;
    m_info.impl().attrs["stride"] = stride;
}

dw::Shape dw::MaxPooling::outShape(const dw::Shape& in_shape) {
    DeepWorks_Assert(in_shape.size() == 4u && "MaxPooling layer works only with 4D tensors");
    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    int h_out = (in_shape[Input::H] + 2 * padding[Kernel::KH] - kernel[Kernel::KH]) / stride[Kernel::KH] + 1;
    int w_out = (in_shape[Input::W] + 2 * padding[Kernel::KW] - kernel[Kernel::KW]) / stride[Kernel::KW] + 1;
    return {in_shape[0], in_shape[1], h_out, w_out};
}

dw::Convolution::Convolution(int out_channels,
                                    const std::array<int, 2>& kernel,
                                    const std::array<int, 2>& padding,
                                    const std::array<int, 2>& stride,
                                    std::string name)
    : BaseOp<dw::Convolution>(LayerInfo(std::move(name), "Convolution")) {
    m_info.impl().attrs["out_channels"] = out_channels;
    m_info.impl().attrs["kernel"] = kernel;
    m_info.impl().attrs["padding"] = padding;
    m_info.impl().attrs["stride"] = stride;
}

void dw::Convolution::init(const Shape& in_shape) {
    int out_channels = m_info.impl().attrs["out_channels"].get<int>();
    auto kernel = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();

    // NB: Init weight.
    m_info.impl().params.emplace("weight", dw::Tensor::xavierUniform({out_channels,
                                                                      in_shape[Input::C],
                                                                      kernel[Kernel::KH],
                                                                      kernel[Kernel::KW]}));
    // NB: Init bias.
    m_info.impl().params.emplace("bias", dw::Tensor::zeros({out_channels}));
}

dw::Shape dw::Convolution::outShape(const dw::Shape& in_shape) {
    DeepWorks_Assert(in_shape.size() == 4u && "Convolution layer works only with 4D tensors");
    int out_channels = m_info.impl().attrs["out_channels"].get<int>();
    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    int h_out = (in_shape[Input::H] + 2 * padding[Kernel::KH] - kernel[Kernel::KH]) / stride[Kernel::KH] + 1;
    int w_out = (in_shape[Input::W] + 2 * padding[Kernel::KW] - kernel[Kernel::KW]) / stride[Kernel::KW] + 1;
    return {in_shape[0], out_channels, h_out, w_out};
}
