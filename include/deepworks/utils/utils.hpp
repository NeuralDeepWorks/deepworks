#pragma once

#include <deepworks/tensor.hpp>

namespace deepworks {
namespace utils {

// The helper function is used to change layout from NCHW to NHWC
void NCHW2NHWC(const Tensor& input, Tensor& output);

// The helper function is used to change layout from NHWC to NCHW
void NHWC2NCHW(const Tensor& input, Tensor& output);

} // namespace utils
} // namespace deepworks
