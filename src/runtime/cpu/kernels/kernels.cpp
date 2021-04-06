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

void deepworks::CPUReLUForward(ConstVector X, Vector result) {
    result = (X.array() > 0.f).select(X, 0.f);
}

void deepworks::CPUReLUInputGrad(ConstVector input,
                                 ConstVector grad_output,
                                 Vector grad_input) {
    grad_input = (input.array() > 0.f).select(grad_output, 0.f);
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
                                      bool is_training, float eps, float alpha,
                                      ConstVector gamma, ConstVector beta) {

    if (is_training) {
        auto input_mean = input.colwise().mean();
        input_centered = input.rowwise() - input_mean;

        auto input_var = input_centered.cwiseAbs2().colwise().mean();
        std = (input_var.array() + eps).cwiseSqrt();

        running_mean = running_mean * alpha + input_mean * (1 - alpha);
        running_var = running_var * alpha + input_var * (1 - alpha);
    } else {
        input_centered = input.rowwise() - running_mean;
        std = (running_var.array() + eps).cwiseSqrt();
    }

    output = (input_centered.array().rowwise() * (gamma.array() / std.array())).array().rowwise() + beta.array();
}

void deepworks::CPUBatchNorm1DInputGrad(ConstMatrix input_centered, ConstVector std,
                                        ConstMatrix grad_output, Matrix grad_input,
                                        ConstVector gamma) {
    auto batch_size = input_centered.outerSize();

    auto grad_x_norm = grad_output.array().rowwise() * gamma.array();

    auto grad_std = (((input_centered.array() * grad_x_norm.array()).array().rowwise() / std.cwiseAbs2().array())
                    .colwise().sum()).array() * (-1.0);

    auto grad_var = grad_std.array() / (std.array() * 2.0).array();

    auto grad_x_centered = (grad_x_norm.array().rowwise() / std.array()) +
                           ((input_centered.array().rowwise() * grad_var.array()).array() * (2.0 / batch_size));

    auto grad_mu = grad_x_centered.colwise().sum();

    grad_input = grad_x_centered.rowwise() - (grad_mu.array() / batch_size).array();
}

void deepworks::CPUBatchNorm1DParamGrad(ConstMatrix input_centered, ConstVector std, ConstMatrix grad_output,
                                        Vector gamma_grad, Vector beta_grad) {

    beta_grad = grad_output.colwise().sum();
    gamma_grad = ((input_centered.array().rowwise() / std.array()).array() * grad_output.array()).colwise().sum();
}
