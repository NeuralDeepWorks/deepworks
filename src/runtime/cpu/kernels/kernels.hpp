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
 * according this formula: output = X * W
 * X have size [batch_size, in_features]
 * W have size [in_features, out_features]
 * result have size [batch_size, out_features]
*/
void CPULinearForward(ConstMatrix X, ConstMatrix W, Matrix result);

/*
 * CPULinearAddBias:
 * implements add bias to output after CPULinearForward
 * b have size [1, out_features]
 * result have size [batch_size, out_features]
*/
void CPULinearAddBias(ConstVector b, Matrix result);

/*
 * CPULinearBackward
 * implements computation backward pass of connected layer
 * input have size [batch_size, in_features]
 * W have size [in_features, out_features]
 * dx have size [batch_size, out_features]
 * dW have size [in_features, out_features]
 * grad_output have size [batch_size, in_features]
*/
void CPULinearBackward(ConstMatrix input, ConstMatrix W, ConstMatrix dx,
                       Matrix dW, Matrix grad_output);

/*
 * CPULinearBiasBackward
 * implements computation backward pass for bias derivative
 * dx have size [batch_size, out_features]
 * db have size [out_features, 1]
*/
void CPULinearBiasBackward(ConstMatrix dx, Vector db);

/*
 * CPUSoftmaxForward
 * implements computation of softmax layer output
 * X have size [batch_size, in_features]
 * result have size [batch_size, in_features]
*/
void CPUSoftmaxForward(ConstMatrix X, Matrix result);

/*
 * CPUSoftmaxBackward
 * implements computation backward pass of softmax layer
 * dx have size [batch_size, in_features]
 * output(after softmax in forward pass) have size [batch_size, in_features]
 * grad_output have size [batch_size, in_features]
*/
void CPUSoftmaxBackward(ConstMatrix dx, ConstMatrix output, Matrix grad_output);

/*
 * CPUReluForward
 * implements computation of relu layer output
 * X have size [batch_size, in_features]
 * result have size [batch_size, in_features]
*/
void CPUReLUForward(ConstMatrix X, Matrix result);

/*
 * CPUReluBackward
 * Implements computation backward pass of relu layer
 * dx have size [batch_size, in_features]
 * output(after relu in forward pass) have size [batch_size, in_features]
 * grad_output have size [batch_size, in_features]
*/
void CPUReLUBackward(ConstMatrix dx, ConstMatrix output, Matrix grad_output);

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

/*
 * CPUBatchNormForward
 * Implements Batch Normalization forward pass
 * input have size [batch_size, in_features]
 * output have size [batch_size, in_features]
 * input_centered have size [batch_size, in_features]
 * std have size [1, in_features]
 * running_mean have size [1, in_features]
 * running_var have size [1, in_features]
 * gamma have size [1, in_features]
 * beta have size [1, in_features]
 */
void CPUBatchNormForward(ConstMatrix input, Matrix output,
                         Matrix input_centered, Vector std,
                         Vector running_mean, Vector running_var,
                         bool isTraining, float alpha,
                         ConstVector gamma, ConstVector beta);

/*
 * CPUBatchNormBackward
 * Implements Batch Normalization forward pass
 * input_centered have size [batch_size, in_features]
 * std have size [1, in_features]
 * grad_output have size [batch_size, in_features]
 * grad_input have size [batch_size, in_features]
 * gamma have size [1, in_features]
 * gamma_grad have size [1, in_features]
 * beta_grad have size [1, in_features]
 */
void CPUBatchNormBackward(ConstMatrix input_centered, ConstVector std,
                          ConstMatrix grad_output, Matrix grad_input,
                          ConstVector gamma, Vector gamma_grad, Vector beta_grad);

} // namespace deepworks