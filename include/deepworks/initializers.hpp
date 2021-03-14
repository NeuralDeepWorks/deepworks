#pragma once

#include <deepworks/tensor.hpp>

namespace deepworks {
namespace initializer {

void zeros(Tensor& tensor);

void constant(Tensor& tensor, float value);

void xavierUniform(Tensor& tensor);

void uniform(Tensor& tensor, float lower = 0.f, float upper = 1.f);

} // namespace initializer
} // namespace deepworks
