#include <vector>

#include <Eigen/Core>
#include <deepworks/loss.hpp>


using ConstMatrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using Matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

using ConstVector = Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;
using Vector = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;


namespace deepworks {
namespace losses {

std::vector<int> GetTargetIndices(const ConstVector target, int rows, int cols) {
    std::vector<int> slice(rows);
    for (int i = 0; i < rows; ++i) {
        slice[i] = static_cast<int>(target(0, i)) + cols * i;
    }
    return slice;
}

float CPUNLLLoss(Matrix X, const ConstVector target) {
    int rows = X.rows();
    int cols = X.cols();

    Vector X_1d(X.data(), X.size());

    std::vector<int> slice = GetTargetIndices(target, rows, cols);

    float loss = -X_1d(0, slice).array().sum() / static_cast<float>(rows);
    return loss;
}

void CPULog(const ConstMatrix X, Matrix LogX) {
    LogX.array() = X.array().log();
}

float CPUCrossEntropyLossForward(const Tensor& X, const Tensor& target) {
    const auto &shape = X.shape();

    int rows = shape[0];
    int cols = shape[1];

    ConstMatrix prob_mat(X.data(), rows, cols);
    ConstVector target_vec(target.data(), rows);

    Eigen::MatrixXf log_prob(rows, cols);
    Matrix log_prob_mat(log_prob.data(), rows, cols);

    CPULog(prob_mat, log_prob_mat);

    float loss = CPUNLLLoss(log_prob_mat, target_vec);
    return loss;
}


void CPUCrossEntropyLossBackward(const Tensor& X, const Tensor& target, Tensor& grad_output) {
    const auto &shape = X.shape();

    int rows = shape[0];
    int cols = shape[1];

    ConstMatrix prob_mat(X.data(), rows, cols);
    ConstVector target_vec(target.data(), rows);
    Matrix grad_output_mat(grad_output.data(), rows, cols);

    grad_output_mat.array() = prob_mat.array();
    Vector grad_output_1d(grad_output_mat.data(), grad_output_mat.size());

    std::vector<int> slice = GetTargetIndices(target_vec, rows, cols);

    grad_output_1d(0, slice).array() -= 1.0;

    grad_output_1d /= static_cast<float>(rows);
}

} // namespace losses
} // namespace deepworks
