#include "runtime/cpu/kernels/kernels.hpp"

// FIXME: It can something general like:
// CPUMul(ConstMatrix X1, ConstMatrix X2, Matrix R)
void deepworks::CPULinearForward(ConstMatrix X, ConstMatrix W, Matrix result) {

    result = X * W;
}

// FIXME: It can something general like:
// CPUAddV(ConstMatrix X, ConstVector V, Matrix R)
void deepworks::CPULinearAddBias(ConstVector b, Matrix result) {

    result = result.rowwise() + b;
}

// FIXME: It can something general like:
// CPUMulBackward.
void deepworks::CPULinearBackward(ConstMatrix input, ConstMatrix W, ConstMatrix dx,
                                  Matrix dW, Matrix grad_output) {

    int batch_size = input.outerSize();
    dW = input.transpose() * dx / batch_size;

    grad_output = dx * W.transpose();
}

// FIXME: It can something general like:
// CPUAddVBackward.
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

void deepworks::CPUBatchNorm1DForward(ConstMatrix input, Matrix output,
                                      Matrix input_centered, Vector std,
                                      Vector running_mean, Vector running_var,
                                      bool isTraining, float eps, float alpha,
                                      ConstVector gamma, ConstVector beta) {

    if (isTraining) {
        auto input_mean = input.colwise().mean();
        input_centered = input.rowwise() - input_mean;

        auto input_var = input_centered.cwiseAbs2().colwise().mean();
        std = (input_var.array() + eps).cwiseSqrt();

        running_mean = running_mean * alpha + input_mean * (1 - alpha);
        running_var = running_var * alpha + input_var * (1 - alpha);

        // FIXME: holy shit, if someone knows how to do it beautifully, tell me
        output = ((input_centered.array().rowwise() / std.array())
                          .array().rowwise() * gamma.array()).array().rowwise() + beta.array();

    } else {
        // FIXME: write in local variable in eval mod normal?
        auto input_centered = input.rowwise() - running_mean;
        auto std = (running_var.array() + eps).cwiseSqrt();

        output = ((input_centered.array().rowwise() / std).array().rowwise() * gamma.array()).array().rowwise() + beta.array();
    }
}

void deepworks::CPUBatchNorm1DBackward(ConstMatrix input_centered, ConstVector std,
                                       ConstMatrix grad_output, Matrix grad_input,
                                       ConstVector gamma, Vector gamma_grad, Vector beta_grad) {

    auto batch_size = input_centered.outerSize();

    beta_grad = grad_output.colwise().sum();
    gamma_grad = ((input_centered.array().rowwise() / std.array()).array() * grad_output.array()).colwise().sum();

    auto grad_x_norm = grad_output.array().rowwise() * gamma.array();

    auto grad_std = (((input_centered.array() * grad_x_norm.array()).array().rowwise() / std.cwiseAbs2().array())
            .colwise().sum()).array() * (-1.0);

    auto grad_var = grad_std.array() / (std.array() * 2).array();

    auto grad_x_centered = (grad_x_norm.array().rowwise() / std.array()) +
                           ((input_centered.array().rowwise() * grad_var.array()).array() * (2.0 / batch_size));

    auto grad_mu = grad_x_centered.colwise().sum();

    grad_input = grad_x_centered.rowwise() - (grad_mu.array() / batch_size).array();
}