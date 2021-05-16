#pragma once

#include <deepworks/placeholder.hpp>
#include <deepworks/base_op.hpp>

namespace deepworks {

Placeholder make_layer(const std::string & name,
                       const std::string & type,
                       const Attributes  & attrs,
                       const Placeholders& inputs);

struct Linear : BaseOp<Linear> {
    Linear(int units, std::string name = "");
    Shape outShape(const Shape& in_shape);
    void init(const Shape& in_shape);
};

struct ReLU : BaseOp<ReLU> {
    ReLU(std::string name = "");
    Shape outShape(const Shape& in_shape);
};

struct Softmax : BaseOp<Softmax> {
    Softmax(std::string name = "");
    Shape outShape(const Shape& in_shape);
};

struct BatchNorm1D : BaseOp<BatchNorm1D> {
    BatchNorm1D(float eps = 1e-5, float alpha = 0.1f, std::string name = "");
    Shape outShape(const Shape& in_shape);
    void init(const Shape& in_shape);
};

struct ELU : BaseOp<ELU> {
    ELU(float alpha, std::string name = "");
    Shape outShape(const Shape& in_shape);
};

struct MaxPooling : BaseOp<MaxPooling> {
    MaxPooling(const std::array<int, 2>& kernel,
               const std::array<int, 2>& padding,
               const std::array<int, 2>& stride,
               std::string name = "");
    Shape outShape(const Shape& in_shape);
};

struct Convolution : BaseOp<Convolution> {
    Convolution(int out_channels,
                const std::array<int, 2>& kernel,
                const std::array<int, 2>& padding,
                const std::array<int, 2>& stride,
                std::string name = "");
    Shape outShape(const Shape& in_shape);
    void init(const Shape& in_shape);
};

struct LeakyReLU : BaseOp<LeakyReLU> {
    LeakyReLU(float alpha, std::string name = "");
    Shape outShape(const Shape& in_shape);
};

struct Sigmoid : BaseOp<Sigmoid> {
    Sigmoid(std::string name = "");
    Shape outShape(const Shape& in_shape);
};

struct Dropout : BaseOp<Dropout> {
    Dropout(float p, std::string name = "");
    void init(const Shape& in_shape);
    Shape outShape(const Shape& in_shape);
};

struct GlobalAvgPooling : BaseOp<GlobalAvgPooling> {
    GlobalAvgPooling(std::string name = "");
    Shape outShape(const Shape& in_shape);
};

struct BatchNorm2D : BaseOp<BatchNorm2D> {
    BatchNorm2D(float eps = 1e-5, float alpha = 0.1f, std::string name = "");
    Shape outShape(const Shape& in_shape);
    void init(const Shape& in_shape);
};

struct Add : BaseOp<Add> {
    Add(std::string name = "");
    Shape outShape(const Shape& lhs_shape, const Shape& rhs_shape);
};

} // namespace deepworks
