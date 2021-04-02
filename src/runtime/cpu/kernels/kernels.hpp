#pragma once

#include <Eigen/Core>

namespace deepworks {

using ConstMatrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using Matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

using ConstVector = Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;
using Vector = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;

/*
 * CPULinearForward:
 * implements computation of fully connected layer output
 * according this formula: output = X * W.T
 * X have size [batch_size, in_features]
 * W have size [out_features, in_features]
 * result have size [batch_size, out_features]
 */
void CPULinearForward(ConstMatrix X, ConstMatrix W, Matrix result);

/*
 * CPULinearAddBias:
 * implements add bias to output after CPULinearForward
 * b have size [1, out_features]
 * result have size [batch_size, out_features]
 */
void CPULinearAddBias(ConstMatrix X, ConstVector b, Matrix result);

/*
 * CPULinearInputGrad
 * Calculates gradients by input for a linear layer
 * dx have size [batch_size, in_features]
 * W have size [out_features, in_features]
 * grad_input have size [batch_size, in_features]
*/
void CPULinearInputGrad(ConstMatrix dx, ConstMatrix W, Matrix grad_input);

/*
 * CPULinearWeightGrad
 * Calculates gradients by weight for a linear layer
 * input have size [batch_size, in_features]
 * dx have size [batch_size, out_features]
 * W have size [out_features, in_features]
 * grad_input have size [batch_size, in_features]
*/
void CPULinearWeightGrad(ConstMatrix input, ConstMatrix dx, Matrix dW);

/*
 * CPULinearBiasGrad
 * Calculates gradients by bias for a linear layer
 * dx have size [batch_size, out_features]
 * db have size [out_features, 1]
*/
void CPULinearBiasGrad(ConstMatrix dx, Vector db);

/*
 * CPUSoftmaxForward
 * implements computation of softmax layer output
 * X have size [batch_size, in_features]
 * result have size [batch_size, in_features]
*/
void CPUSoftmaxForward(ConstMatrix X, Matrix result);

/*
 * CPUSoftmaxInputGrad
 * Calculates gradients by input for a softmax layer
 * output (after softmax in forward pass) have size [batch_size, in_features]
 * grad_output have size [batch_size, in_features]
 * grad_input have size [batch_size, in_features]
*/
void CPUSoftmaxInputGrad(ConstMatrix output, ConstMatrix grad_output, Matrix grad_input);

/*
 * CPUReluForward
 * implements computation of relu layer output
 * X is a s 1D vector [size]
 * result have size [size]
*/
void CPUReLUForward(ConstVector X, Vector result);

/*
 * CPUReLUInputGrad
 * Calculates gradients by input for a relu layer
 * input is a 1D vector [size]
 * grad_output have size [size]
 * grad_input have size [size]
*/
void CPUReLUInputGrad(ConstVector input, ConstVector grad_output, Vector grad_input);

/*
 * CPULog
 * Implements application Log to X, saves it to LogX
 * X have size [batch_size, in_features]
 * LogX have size [batch_size, in_features]
 */
void CPULog(ConstMatrix X, Matrix LogX);

/*
 * MatchTargetTo1dMatrix
 * Implements easy access to the required parameters in the predictions matrix
 * target have size [1, batch_size], where values are in the range [0, N_classes-1]
 */
std::vector<int> MatchTargetTo1dMatrix(ConstVector target, int batch_size, int n_classes);

/*
 * CPUNLLLoss
 * Implements classic NLLLoss
 * predictions have size [batch_size, N_classes]
 * target have size [1, batch_size], where values are in the range [0, N_classes-1]
 */
float CPUNLLLoss(Matrix predictions, ConstVector target);

} // namespace deepworks
