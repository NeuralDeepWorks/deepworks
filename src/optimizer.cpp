#include <Eigen/Core>

#include <deepworks/optimizer.hpp>
#include <util/assert.hpp>

namespace deepworks {
namespace optimizer {

using ConstMatrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using Matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

SGD::SGD(Parameters& params, float lr) : m_params(params), m_lr(lr) {}

void SGD::step() {
    for (auto& param: m_params) {
        if (param.is_trainable()) {
            auto weight = param.data();
            const auto grad = param.grad();

            const auto& shape = weight.shape();

            DeepWorks_Assert(grad.shape() == shape);
            DeepWorks_Assert(shape.size() == 2 || shape.size() == 1);

            int rows = shape[0];
            int cols = shape.size() == 2 ? shape[1] : 1;

            Matrix weight_mat(weight.data(), rows, cols);
            ConstMatrix grad_mat(grad.data(), rows, cols);

            weight_mat.array() -= m_lr * grad_mat.array();
        }
    }
}

float SGD::get_lr() const {
    return m_lr;
}

void SGD::set_lr(float lr) {
    m_lr = lr;
}

} // namespace optimizer
} // namespace deepworks
