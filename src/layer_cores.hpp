#pragma once

#include <Eigen/Core>

namespace deepworks {

    /*
     * CPULinearForward:
     * implements computation of fully connected layer output
     * according this formula: output = X * W + b
     * X have size [batch_size, in_features]
     * W have size [in_features, out_features]
     * b have size [out_features, 1]
     * result have size [batch_size, out_features]
    */
    void CPULinearForward(const Eigen::MatrixXf & X, const Eigen::MatrixXf & W,
                          const Eigen::VectorXf & b, Eigen::MatrixXf & result);

    /*
     * CPULinearBackward
     * implements computation backward pass of connected layer
     * input have size [batch_size, in_features]
     * W have size [in_features, out_features]
     * dx have size [batch_size, out_features]
     * dW have size [in_features, out_features]
     * db have size [out_features, 1]
     * grad_output have size [batch_size, in_features]
    */
    void CPULinearBackward(const Eigen::MatrixXf & input, const Eigen::MatrixXf & W, const Eigen::MatrixXf & dx,
                           Eigen::MatrixXf & dW, Eigen::VectorXf & db, Eigen::MatrixXf & grad_output);

    /*
     * CPUSoftmaxForward
     * implements computation of softmax layer output
     * X have size [batch_size, in_features]
     * result have size [batch_size, in_features]
    */
    void CPUSoftmaxForward(const Eigen::MatrixXf & X, Eigen::MatrixXf & result);

    /*
     * CPUSoftmaxBackward
     * implements computation backward pass of softmax layer
     * dx have size [batch_size, in_features]
     * output(after softmax in forward pass) have size [batch_size, in_features]
     * grad_output have size [batch_size, in_features]
    */
    void CPUSoftmaxBackward(const Eigen::MatrixXf & dx, const Eigen::MatrixXf & output, Eigen::MatrixXf & grad_output);

    /*
     * CPUReluForward
     * implements computation of relu layer output
     * X have size [batch_size, in_features]
     * result have size [batch_size, in_features]
    */
    void CPUReluForward(const Eigen::MatrixXf & X, Eigen::MatrixXf & result);

    /*
     * CPUReluBackward
     * Implements computation backward pass of relu layer
     * dx have size [batch_size, in_features]
     * output(after relu in forward pass) have size [batch_size, in_features]
     * grad_output have size [batch_size, in_features]
    */
    void CPUReluBackward(const Eigen::MatrixXf & dx, const Eigen::MatrixXf & output, Eigen::MatrixXf & grad_output);

} // namespace deepworks
