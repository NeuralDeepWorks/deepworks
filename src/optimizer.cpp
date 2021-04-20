#include <Eigen/Core>

#include <deepworks/optimizer.hpp>
#include <util/assert.hpp>

namespace deepworks {
namespace optimizer {

using ConstVector = Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;
using Vector = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;

SGD::SGD(ParamMap& params, float lr) : m_params(params), m_lr(lr) {}

void SGD::step() {
    for (auto& [name, param]: m_params) {
        if (param.is_trainable()) {
            auto weight = param.data();
            const auto grad = param.grad();

            DeepWorks_Assert(grad.shape() == weight.shape());

            const size_t size = grad.total();

            Vector weight_mat(weight.data(), size);
            ConstVector grad_mat(grad.data(), size);

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
