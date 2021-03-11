#pragma once

#include <tensor.hpp>

namespace deepworks {
namespace metric {

float accuracy(const Tensor& y_pred, const Tensor& y_true);

float sparse_accuracy(const Tensor& y_pred, const Tensor& y_true);

} // namespace metric
} // namespace deepworks
