#include "layer_cores.hpp"

void CPULinearForward(const Eigen::MatrixXf & X, const Eigen::MatrixXf & W,
                      const Eigen::VectorXf & b, Eigen::MatrixXf & result) {

    result = (X * W).rowwise() + b.transpose();
}

void CPULinearBackward(const Eigen::MatrixXf & input, const Eigen::MatrixXf & W, const Eigen::MatrixXf & dx,
                       Eigen::MatrixXf & dW, Eigen::VectorXf & db, Eigen::MatrixXf & grad_output) {

    int batch_size = input.innerSize();
    dW = input.transpose() * dx / batch_size;
    db = (dx.colwise().sum() / batch_size).transpose();

    grad_output = dx * W.transpose();
}

void CPUSoftmaxForward(const Eigen::MatrixXf & X, Eigen::MatrixXf & result) {

    Eigen::MatrixXf exp_x = (X.colwise() - X.rowwise().maxCoeff()).array().exp();
    result = exp_x.array().colwise() / exp_x.rowwise().sum().array();
}

void CPUSoftmaxBackward(const Eigen::MatrixXf & dx, const Eigen::MatrixXf & output, Eigen::MatrixXf & grad_output) {

    Eigen::VectorXf k = (dx.array() * output.array()).rowwise().sum();
    grad_output = output.array() * (dx.colwise() - k).array();
}

void CPUReluForward(const Eigen::MatrixXf & X, Eigen::MatrixXf & result) {

    result = (X.array() < 0).select(0, X);
}

void CPUReluBackward(const Eigen::MatrixXf & dx, const Eigen::MatrixXf & output, Eigen::MatrixXf & grad_output) {

    Eigen::MatrixXf ones = Eigen::MatrixXf::Ones(dx.innerSize(), dx.outerSize());
    Eigen::MatrixXf mask = (output.array() > 0).select(ones, 0.0);
    grad_output = mask.array() * dx.array();
}
