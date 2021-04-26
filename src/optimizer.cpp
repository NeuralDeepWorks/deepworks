#include <Eigen/Core>

#include <deepworks/initializers.hpp>
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

SGDMomentum::SGDMomentum(ParamMap& params, float lr, float gamma) : m_params(params), m_lr(lr), m_gamma(gamma) {
    for (auto& [name, param]: m_params) {
        m_velocities.emplace(name, Tensor::zeros(param.data().shape()));
    }
}

void SGDMomentum::step() {
    for (auto& [name, param] : m_params) {
        if (param.is_trainable()) {
            auto       weight = param.data();
            const auto grad   = param.grad();

            DeepWorks_Assert(grad.shape() == weight.shape());

            const size_t size = grad.total();

            Vector      velocity_mat(m_velocities.at(name).data(), size);
            Vector      weight_mat(weight.data(), size);
            ConstVector grad_mat(grad.data(), size);

            velocity_mat.array() = m_gamma * velocity_mat.array() + m_lr * grad_mat.array();

            weight_mat.array() -= velocity_mat.array();
        }
    }
}

float SGDMomentum::get_lr() const {
    return m_lr;
}

void SGDMomentum::set_lr(float lr) {
    m_lr = lr;
}

} // namespace optimizer
} // namespace deepworks
