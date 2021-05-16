#pragma once

#include <Eigen/Core>
#include <deepworks/tensor.hpp>

namespace deepworks {

using ConstMatrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using Matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

using ConstVector = Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;
using Vector = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;

using ConstColVector = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor>>;
using ColVector = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor>>;

/*
 * CPUAddForward:
 * Computes sum of two matrices
 * X1 and X2 have the same size
 * result have the same size as inputs
 */
void CPUAddForward(const Tensor& X1, const Tensor& X2, Tensor& result);

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
 * X is a 1D vector [size]
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
 * CPULeakyReLUForward
 * implements computation of LeakyReLU layer output,
 * according this formula:
 *
 *                / x, if x > 0
 * LeakyReLU(x) = |
 *                \ alpha * x, if x <= 0
 *
 * X is a 1D vector [size]
 * result have size [size]
 * alpha coefficient from formula above
*/
void CPULeakyReLUForward(ConstVector X, Vector result, float alpha);

/*
 * CPULeakyReLUInputGrad
 * Calculates gradients by input for a LeakyReLU layer
 * input is a 1D vector [size]
 * grad_output have size [size]
 * grad_input have size [size]
*/
void CPULeakyReLUInputGrad(ConstVector input, ConstVector grad_output, Vector grad_input, float alpha);

/*
 * CPUELUForward
 * implements computation of ELU layer output,
 * according this formula:
 *
 *          / x, if x > 0
 * ELU(x) = |
 *          \ alpha * (e^x - 1), if x < 0
 *
 * X is a 1D vector [size]
 * result have size [size]
 * alpha coefficient from formula above
*/
void CPUELUForward(ConstVector X, Vector result, float alpha);

/*
 * CPUELUInputGrad
 * Calculates gradients by input for a ELU layer
 * input is a 1D vector [size]
 * grad_output have size [size]
 * grad_input have size [size]
*/
void CPUELUInputGrad(ConstVector input, ConstVector grad_output, Vector grad_input, float alpha);

/*
 * CPUSigmoidorward
 * implements computation of sigmoid layer output
 * X is a 1D vector [size]
 * result have size [size]
*/
void CPUSigmoidForward(ConstVector X, Vector result);

/*
 * CPUSigmoidInputGrad
 * Calculates gradients by input for a sigmoid layer
 * output is a 1D vector [size]
 * grad_output have size [size]
 * grad_input have size [size]
*/
void CPUSigmoidInputGrad(ConstVector output, ConstVector grad_output, Vector grad_input);

/*
 * CPUConvolutionalForward
 * implements computation of convolution layer output
 * input has size [batch_size, c_in, height_in, width_in]
 * weights has size [c_out, c_in, kernel_h, kernel_w]
 * bias has size [c_out]
 * output has size [batch_size, c_out, height_out, width_out]
 * im2col_buf helper tensor stores im2col data representation
 * kernel stores [kernel_h, kernel_w]
 * padding stores [padding_h, padding_w]
 * stride stores [stride_h, stride_w]
*/
void CPUConvolutionalForward(const Tensor& input,
                             const Tensor& weights,
                             const Tensor& bias,
                             Tensor& output,
                             Tensor& im2col_buf,
                             const std::array<int, 2>& kernel,
                             const std::array<int, 2>& padding,
                             const std::array<int, 2>& stride);

/*
 * CPUConvolutionalInputGrad
 * Implements backward pass of convolution layer
 * grad_output has size [batch_size, c_out, height_out, width_out]
 * weights has size [c_out, c_in, kernel_h, kernel_w]
 * im2col_buf helper tensor stores im2col data representation
 * grad_input has size [batch_size, c_in, height_in, width_in]
 * kernel stores [kernel_h, kernel_w]
 * padding stores [padding_h, padding_w]
 * stride stores [stride_h, stride_w]
*/
void CPUConvolutionalInputGrad(const Tensor& grad_output,
                               const Tensor& weights,
                               const Tensor& im2col_buf,
                               Tensor& grad_input,
                               const std::array<int, 2>& kernel,
                               const std::array<int, 2>& padding,
                               const std::array<int, 2>& stride);

/*
 * CPUConvolutionalWeightsGrad
 * Implements backward pass of convilution layer
 * grad_output has size [batch_size, c_out, height_out, width_out]
 * im2col_buf helper tensor stores im2col data representation
 * grad_weights has size [c_out, c_in, kernel_h, kernel_w]
*/
void CPUConvolutionalWeightsGrad(const Tensor& grad_output,
                                 const Tensor& im2col_buf,
                                 Tensor& grad_weights);

/*
 * CPUConvolutionalBiasGrad
 * Implements backward pass of convilution layer
 * grad_output has size [batch_size, c_out, height_out, width_out]
 * grad_bias has size [c_out]
*/
void CPUConvolutionalBiasGrad(const Tensor& grad_output, Tensor& grad_bias);

/*
 * CPUMaxPoolingForward
 * Implements forward pass of max pooling layer
 * input has size [batch_size, channels, height_in, width_in]
 * max_indices stores the max element indexes for each kernel, has the same size as output
 * output has size [batch_size, channels, height_out, width_out]
 * kernel stores [kernel_h, kernel_w]
 * padding stores [padding_h, padding_w]
 * stride stores [stride_h, stride_w]
*/
void CPUMaxPoolingForward(const Tensor& input,
                          Tensor& max_indices,
                          Tensor& im2col_buf,
                          Tensor& output,
                          const std::array<int, 2>& kernel,
                          const std::array<int, 2>& padding,
                          const std::array<int, 2>& stride);

/*
 * CPUMaxPoolingInputGrad
 * Implements backward pass of max pooling layer
 * grad_output has size [batch_size, channels, height_out, width_out]
 * max_indices stores the max element indexes for each kernel, has the same size as grad_output
 * grad_input has size [batch_size, channels, height_in, width_in]
 * kernel stores [kernel_h, kernel_w]
 * padding stores [padding_h, padding_w]
 * stride stores [stride_h, stride_w]
*/
void CPUMaxPoolingInputGrad(const Tensor& grad_output,
                            const Tensor& max_indices,
                            Tensor& im2col_buf,
                            Tensor& grad_input,
                            const std::array<int, 2>& kernel,
                            const std::array<int, 2>& padding,
                            const std::array<int, 2>& stride);

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
 * CPUBatchNorm1DForward
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
void CPUBatchNorm1DForward(ConstMatrix input, Matrix output,
                           Matrix input_centered, Vector std,
                           Vector running_mean, Vector running_var,
                           bool is_training, float eps, float alpha,
                           ConstVector gamma, ConstVector beta);

/*
 * CPUBatchNorm1DInputGrad
 * Implements Batch Normalization backward pass for input
 * input_centered have size [batch_size, in_features]
 * std have size [1, in_features]
 * grad_output have size [batch_size, in_features]
 * grad_input have size [batch_size, in_features]
 * gamma have size [1, in_features]
 */
void CPUBatchNorm1DInputGrad(ConstMatrix input_centered, ConstVector std,
                             ConstMatrix grad_output, Matrix grad_input,
                             ConstVector gamma);

/*
 * CPUBatchNorm1DParamGrad
 * Implements Batch Normalization backward pass for params
 * input_centered have size [batch_size, in_features]
 * std have size [1, in_features]
 * grad_output have size [batch_size, in_features]
 * gamma_grad have size [1, in_features]
 * beta_grad have size [1, in_features]
 */
void CPUBatchNorm1DParamGrad(ConstMatrix input_centered, ConstVector std, ConstMatrix grad_output,
                             Vector gamma_grad, Vector beta_grad);

/*
 * CPUDropoutForward
 * Implements Dropout forward pass
 * input  - input tensor
 * mask   - tensor with probabilites
 * output - output tensor with the same size as input
 * p      - probability of an element to be zeroed
 */
void CPUDropoutForward(const Tensor& input, Tensor& mask, Tensor& output, float p);

 /*
 * CPUDropoutInputGrad
 * Implements Dropout backward pass
 * mask        - tensor with probabilites
 * grad_output - tensor with output gradient
 * grad_input  - tensor with input gradient
 * p           - probability of an element to be zeroed
 */
void CPUDropoutInputGrad(const Tensor& mask, const Tensor& grad_output, Tensor& grad_input, float p);

/*
 * CPUGlobalAvgPoolingForward
 * Implements Global Average Pooling forward pass
 * input has size [batch_size, channels, heights, widths]
 * output has size [batch_size, channels, 1, 1]
 */
void CPUGlobalAvgPoolingForward(const Tensor& input, Tensor& output);

/*
 * CPUGlobalAvgPoolingInputGrad
 * Implements Global Average Pooling backward pass
 * grad_output has size [batch_size, channels, 1, 1]
 * grad_input has size [batch_size, channels, heights, widths]
 */
void CPUGlobalAvgPoolingInputGrad(const Tensor& grad_output, Tensor& grad_input);

// The helper function is used in the Convolution and MaxPooling kernels for the forward pass
void im2col(const Tensor& image,
            Tensor& im2col_buf,
            const std::array<int, 2>& kernel,
            const std::array<int, 2>& padding,
            const std::array<int, 2>& stride,
            const std::array<int, 2>& dilation = {1, 1},
            float fill_value = 0.f);

// The helper function is used in the Convolution and MaxPooling kernels for the backward pass
void col2im(const Tensor& im2col_buf,
            Tensor& image,
            const std::array<int, 2>& kernel,
            const std::array<int, 2>& padding,
            const std::array<int, 2>& stride,
            const std::array<int, 2>& dilation = {1, 1});
} // namespace deepworks
