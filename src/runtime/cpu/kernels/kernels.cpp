#include "runtime/cpu/kernels/kernels.hpp"

void deepworks::CPULinearForward(ConstMatrix X, ConstMatrix W, Matrix result) {

    result = X * W;
}

void deepworks::CPULinearAddBias(ConstVector b, Matrix result) {

    result = result.rowwise() + b;
}

void deepworks::CPULinearBackward(ConstMatrix input, ConstMatrix W, ConstMatrix dx,
                                  Matrix dW, Matrix grad_output) {

    int batch_size = input.outerSize();
    dW = input.transpose() * dx / batch_size;

    grad_output = dx * W.transpose();
}

void deepworks::CPULinearBiasBackward(ConstMatrix dx, Vector db) {

    int batch_size = dx.outerSize();
    db = (dx.colwise().sum() / batch_size);
}

void deepworks::CPUSoftmaxForward(ConstMatrix X, Matrix result) {

    Eigen::MatrixXf exp_x = (X.colwise() - X.rowwise().maxCoeff()).array().exp();
    result = exp_x.array().colwise() / exp_x.rowwise().sum().array();
}

void deepworks::CPUSoftmaxBackward(ConstMatrix dx, ConstMatrix output, Matrix grad_output) {

    Eigen::VectorXf k = (dx.array() * output.array()).rowwise().sum();
    grad_output = output.array() * (dx.colwise() - k).array();
}

void deepworks::CPUReLUForward(ConstMatrix X, Matrix result) {

    result = (X.array() > 0).select(X, 0);
}

void deepworks::CPUReLUBackward(ConstMatrix dx, ConstMatrix output, Matrix grad_output) {

    grad_output = (output.array() > 0.0).select(dx, 0.0);
}

void deepworks::CPULog(ConstMatrix X, Matrix LogX) {
    LogX.array() = X.array().log();
}

std::vector<int> deepworks::MatchTargetTo1dMatrix(ConstVector target, int batch_size, int n_classes) {
    std::vector<int> slice(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        slice[i] = static_cast<int>(target(0, i)) + n_classes * i;
    }
    return slice;
}

float deepworks::CPUNLLLoss(Matrix predictions, ConstVector target) {
    int batch_size = predictions.rows();
    int n_classes = predictions.cols();

    Vector X_1d(predictions.data(), predictions.size());

    std::vector<int> slice = MatchTargetTo1dMatrix(target, batch_size, n_classes);

    float loss = -X_1d(0, slice).array().sum() / static_cast<float>(batch_size);
    return loss;
}
