#pragma once

#include <Eigen/Core>

using MatrixMapper = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using VectorMapper = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;

namespace deepworks {
namespace cpu {
namespace kernels {

void CPUReLUForward(const MatrixMapper & X, MatrixMapper & result);

} // namespace kernels
} // namespace cpu
} // namespace deepworks
