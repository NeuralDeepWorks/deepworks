#include <vector>

#include <Eigen/Core>

#include <deepworks/initializers.hpp>
#include <deepworks/loss.hpp>
#include <runtime/cpu/kernels/kernels.hpp>
#include <util/assert.hpp>

namespace deepworks {
namespace loss {

float CrossEntropyLoss::CPUForward(const Tensor& predictions, const Tensor& target) {
    const auto& shape = predictions.shape();
    DeepWorks_Assert(shape.size() == 2);

    int batch_size = shape[0];
    int n_classes = shape[1];

    DeepWorks_Assert(target.shape().size() == 1);
    DeepWorks_Assert(target.shape()[0] == batch_size);

    ConstMatrix prob_mat(predictions.data(), batch_size, n_classes);
    ConstVector target_vec(target.data(), batch_size);

    if (log_predictions.shape() != shape) {
        log_predictions.allocate(shape);
    }
    deepworks::initializer::zeros(log_predictions);

    Matrix log_prob_mat(log_predictions.data(), batch_size, n_classes);

    CPULog(prob_mat, log_prob_mat);

    float loss = CPUNLLLoss(log_prob_mat, target_vec);
    return loss;
}

void CrossEntropyLoss::CPUBackward(const Tensor& predictions, const Tensor& target, Tensor& grad_output) {
    const auto& shape = predictions.shape();
    DeepWorks_Assert(shape.size() == 2);
    DeepWorks_Assert(grad_output.shape() == shape);

    int batch_size = shape[0];
    int n_classes = shape[1];

    DeepWorks_Assert(target.shape().size() == 1);
    DeepWorks_Assert(target.shape()[0] == batch_size);

    ConstMatrix prob_mat(predictions.data(), batch_size, n_classes);
    ConstVector target_vec(target.data(), batch_size);
    Matrix grad_output_mat(grad_output.data(), batch_size, n_classes);

    Vector grad_output_1d(grad_output_mat.data(), grad_output_mat.size());
    ConstVector prob_mat_1d(prob_mat.data(), prob_mat.size());

    std::vector<int> slice = MatchTargetTo1dMatrix(target_vec, batch_size, n_classes);

    grad_output_1d(0, slice).array() = -1 / prob_mat_1d(0, slice).array();

    grad_output_1d /= static_cast<float>(batch_size);
}

} // namespace loss
} // namespace deepworks
