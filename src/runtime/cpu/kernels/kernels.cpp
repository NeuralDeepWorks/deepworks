#include "runtime/cpu/kernels/kernels.hpp"

void deepworks::CPULinearForward(ConstMatrix X, ConstMatrix W, Matrix result) {

    result = X * W.transpose();
}

void deepworks::CPULinearAddBias(ConstMatrix X, ConstVector b, Matrix result) {

    result = X.rowwise() + b;
}

void deepworks::CPULinearInputGrad(ConstMatrix dx,
                                   ConstMatrix W,
                                   Matrix grad_input) {

    grad_input = dx * W;
}

void deepworks::CPULinearWeightGrad(ConstMatrix input, ConstMatrix dx, Matrix dW) {

    dW = dx.transpose() * input;
}

void deepworks::CPULinearBiasGrad(ConstMatrix dx, Vector db) {

    db = dx.colwise().sum();
}

void deepworks::CPUSoftmaxForward(ConstMatrix X, Matrix result) {

    Eigen::MatrixXf exp_x = (X.colwise() - X.rowwise().maxCoeff()).array().exp();
    result = exp_x.array().colwise() / exp_x.rowwise().sum().array();
}

void deepworks::CPUSoftmaxInputGrad(ConstMatrix output, ConstMatrix grad_output, Matrix grad_input) {

    Eigen::VectorXf k = (grad_output.array() * output.array()).rowwise().sum();
    grad_input = output.array() * (grad_output.colwise() - k).array();
}

void deepworks::CPUReLUForward(ConstMatrix X, Matrix result) {

    result = (X.array() > 0.f).select(X, 0.f);
}

void deepworks::CPUReLUInputGrad(ConstMatrix input,
                                 ConstMatrix grad_output,
                                 Matrix grad_input) {

    grad_input = (input.array() > 0.0).select(grad_output, 0.0);
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
