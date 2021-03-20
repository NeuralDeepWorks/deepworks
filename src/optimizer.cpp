#include <deepworks/optimizer.hpp>

namespace deepworks {
namespace optimizer {

using ConstMatrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using Matrix = Eigen::Map <Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

SGD::SGD(int& params, float lr) : parameters(params), learning_rate(lr) {}

void SGD::step() {
    for (auto& parameter: parameters) {
        if (parameter.is_trainable()) {
            auto weight = parameter.data();
            const auto grad = parameter.grad();

            const auto& shape = weight.shape();
            DeepWorks_Assert(shape.size() == 2);
            DeepWorks_Assert(grad.shape() == shape);

            Matrix weight_mat(weight.data(), shape[0], shape[1]);
            ConstMatrix grad_mat(grad.data(), shape[0], shape[1]);

            weight_mat -= learning_rate * grad_mat;
        }
    }
}

float SGD::get_learning_rate() {
    return learning_rate;
}

float SGD::set_learning_rate(float lr) {
    learning_rate = lr;
}

} // namespace optimizer
} // namespace deepworks