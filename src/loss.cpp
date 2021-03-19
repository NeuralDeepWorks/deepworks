#include <vector>

#include <Eigen/Core>
#include <deepworks/loss.hpp>


using ConstMatrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using Matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

using ConstVector = Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;
using Vector = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;


namespace deepworks {
namespace loss {

std::vector<int> GetTargetIndices(const ConstVector target, int batch_size, int n_classes) {
    std::vector<int> slice(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        slice[i] = static_cast<int>(target(0, i)) + n_classes * i;
    }
    return slice;
}

float CPUNLLLoss(Matrix X, const ConstVector target) {
    int batch_size = X.rows();
    int n_classes = X.cols();

    Vector X_1d(X.data(), X.size());

    std::vector<int> slice = GetTargetIndices(target, batch_size, n_classes);

    float loss = -X_1d(0, slice).array().sum() / static_cast<float>(batch_size);
    return loss;
}

void CPULog(const ConstMatrix X, Matrix LogX) {
    LogX.array() = X.array().log();
}

float CPUCrossEntropyLossForward(const Tensor& X, const Tensor& target) {
    const auto &shape = X.shape();

    int batch_size = shape[0];
    int n_classes = shape[1];

    ConstMatrix prob_mat(X.data(), batch_size, n_classes);
    ConstVector target_vec(target.data(), batch_size);

    Eigen::MatrixXf log_prob(batch_size, n_classes);
    Matrix log_prob_mat(log_prob.data(), batch_size, n_classes);

    CPULog(prob_mat, log_prob_mat);

    float loss = CPUNLLLoss(log_prob_mat, target_vec);
    return loss;
}


void CPUCrossEntropyLossBackward(const Tensor& X, const Tensor& target, Tensor& grad_output) {
    const auto &shape = X.shape();

    int batch_size = shape[0];
    int n_classes = shape[1];

    ConstMatrix prob_mat(X.data(), batch_size, n_classes);
    ConstVector target_vec(target.data(), batch_size);
    Matrix grad_output_mat(grad_output.data(), batch_size, n_classes);

    Vector grad_output_1d(grad_output_mat.data(), grad_output_mat.size());
    ConstVector prob_mat_1d(prob_mat.data(), prob_mat.size());

    std::vector<int> slice = GetTargetIndices(target_vec, batch_size, n_classes);

    grad_output_1d(0, slice).array() = -prob_mat_1d(0, slice).array().pow(-1);

    grad_output_1d /= static_cast<float>(batch_size);
}

} // namespace loss
} // namespace deepworks
