#pragma once

#include <deepworks/tensor.hpp>

namespace deepworks {
namespace metric {

float accuracy(const Tensor& y_pred, const Tensor& y_true);

float accuracyOneHot(const Tensor& y_pred, const Tensor& y_true);

} // namespace metric
} // namespace deepworks
