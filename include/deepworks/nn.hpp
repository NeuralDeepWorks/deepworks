#pragma once

#include <deepworks/placeholder.hpp>
#include <deepworks/layer_info.hpp>
#include <deepworks/call.hpp>

namespace deepworks {

template <typename D>
struct BaseOp {
    BaseOp(LayerInfo&& info) : m_info(std::move(info)) { }

    void init(const Shape& in_shape) { /* do nothing */ }

    Placeholder operator()(Placeholder in) {
        static_cast<D*>(this)->init(in.shape());

        Call call{m_info};
        call.pass({in});
        return call.create(static_cast<D*>(this)->outShape(in.shape()));
    }

    LayerInfo m_info;
};

struct Linear : BaseOp<Linear> {
    Linear(int units, std::string name);
    Shape outShape(const Shape& in_shape);
    void init(const Shape& in_shape);
};

struct ReLU : BaseOp<ReLU> {
    ReLU(std::string name);
    Shape outShape(const Shape& in_shape);
};

struct Softmax : BaseOp<Softmax> {
    Softmax(std::string name);
    Shape outShape(const Shape& in_shape);
};

struct BatchNorm1D : BaseOp<BatchNorm1D> {
    BatchNorm1D(float eps, float alpha, std::string name);
    Shape outShape(const Shape& in_shape);
    void init(const Shape& in_shape);
};

struct ELU : BaseOp<ELU> {
    ELU(float alpha, std::string name);
    Shape outShape(const Shape& in_shape);
};

struct MaxPooling : BaseOp<MaxPooling> {
    MaxPooling(const std::array<int, 2>& kernel,
               const std::array<int, 2>& padding,
               const std::array<int, 2>& stride,
               std::string name);
    Shape outShape(const Shape& in_shape);
};

struct Convolution : BaseOp<Convolution> {
    Convolution(int out_channels,
                const std::array<int, 2>& kernel,
                const std::array<int, 2>& padding,
                const std::array<int, 2>& stride,
                std::string name);
    Shape outShape(const Shape& in_shape);
    void init(const Shape& in_shape);
};

struct LeakyReLU : BaseOp<LeakyReLU> {
    LeakyReLU(float alpha, std::string name);
    Shape outShape(const Shape& in_shape);
};

struct Sigmoid : BaseOp<Sigmoid> {
    Sigmoid(std::string name);
    Shape outShape(const Shape& in_shape);
};

struct Dropout : BaseOp<Dropout> {
    Dropout(float p, std::string name);
    void init(const Shape& in_shape);
    Shape outShape(const Shape& in_shape);
};
} // namespace deepworks
