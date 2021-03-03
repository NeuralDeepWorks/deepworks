#include "kernels.hpp"

void deepworks::CPULinearForward(const MatrixMapper& X, const MatrixMapper& W, MatrixMapper& result) {

    result = X * W;
}

void deepworks::CPULinearAddBias(const VectorMapper& b, MatrixMapper& result) {

    result = result.rowwise() + b;
}

void deepworks::CPULinearBackward(const MatrixMapper& input, const MatrixMapper& W, const MatrixMapper& dx,
                                  MatrixMapper& dW, MatrixMapper& grad_output) {

    int batch_size = input.outerSize();
    dW = input.transpose() * dx / batch_size;

    grad_output = dx * W.transpose();
}

void deepworks::CPULinearBiasBackward(const MatrixMapper& dx, VectorMapper& db) {

    int batch_size = dx.outerSize();
    db = (dx.colwise().sum() / batch_size);
}

void deepworks::CPUSoftmaxForward(const MatrixMapper& X, MatrixMapper& result) {

    Eigen::MatrixXf exp_x = (X.colwise() - X.rowwise().maxCoeff()).array().exp();
    result = exp_x.array().colwise() / exp_x.rowwise().sum().array();
}

void deepworks::CPUSoftmaxBackward(const MatrixMapper& dx, const MatrixMapper& output, MatrixMapper& grad_output) {

    Eigen::VectorXf k = (dx.array() * output.array()).rowwise().sum();
    grad_output = output.array() * (dx.colwise() - k).array();
}

void deepworks::CPUReLUForward(const MatrixMapper& X, MatrixMapper& result) {

    result = (X.array() > 0).select(X, 0);
}

void deepworks::CPUReLUBackward(const MatrixMapper& dx, const MatrixMapper& output, MatrixMapper& grad_output) {

    grad_output = (output.array() > 0.0).select(dx, 0.0);
}
