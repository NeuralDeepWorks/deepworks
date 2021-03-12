#include "runtime/cpu/kernels/kernels.hpp"

void deepworks::CPULinearForward(const ConstMatrix& X, const ConstMatrix& W, Matrix& result) {

    result = X * W;
}

void deepworks::CPULinearAddBias(const ConstVector& b, Matrix& result) {

    result = result.rowwise() + b;
}

void deepworks::CPULinearBackward(const ConstMatrix& input, const ConstMatrix& W, const ConstMatrix& dx,
                                  Matrix& dW, Matrix& grad_output) {

    int batch_size = input.outerSize();
    dW = input.transpose() * dx / batch_size;

    grad_output = dx * W.transpose();
}

void deepworks::CPULinearBiasBackward(const ConstMatrix& dx, Vector& db) {

    int batch_size = dx.outerSize();
    db = (dx.colwise().sum() / batch_size);
}

void deepworks::CPUSoftmaxForward(const ConstMatrix& X, Matrix& result) {

    Eigen::MatrixXf exp_x = (X.colwise() - X.rowwise().maxCoeff()).array().exp();
    result = exp_x.array().colwise() / exp_x.rowwise().sum().array();
}

void deepworks::CPUSoftmaxBackward(const ConstMatrix& dx, const ConstMatrix& output, Matrix& grad_output) {

    Eigen::VectorXf k = (dx.array() * output.array()).rowwise().sum();
    grad_output = output.array() * (dx.colwise() - k).array();
}

void deepworks::CPUReLUForward(const ConstMatrix& X, Matrix& result) {

    result = (X.array() > 0).select(X, 0);
}

void deepworks::CPUReLUBackward(const ConstMatrix& dx, const ConstMatrix& output, Matrix& grad_output) {

    grad_output = (output.array() > 0.0).select(dx, 0.0);
}
