#include <Eigen/Core>

#include <deepworks/initializers.hpp>
#include <deepworks/optimizer.hpp>
#include <util/assert.hpp>

namespace deepworks {
namespace optimizer {

using ConstVector = Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;
using Vector = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;

SGD::SGD(Parameters& params, float lr) : m_params(params), m_lr(lr) {}

void SGD::step() {
    for (auto& param: m_params) {
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

Momentum::Momentum(Parameters& params, float lr, float gamma) : m_params(params), m_lr(lr), gamma(gamma) {
    for (auto& param: m_params) {
        velocities.emplace_back(Tensor{param.data().shape()});
        initializer::zeros(velocities.back());
    }
}

void Momentum::step() {
    for (size_t i = 0; i < m_params.size(); ++i) {
        if (m_params[i].is_trainable()) {
            auto       weight = m_params[i].data();
            const auto grad   = m_params[i].grad();

            DeepWorks_Assert(grad.shape() == weight.shape());

            const size_t size = grad.total();

            Vector      velocity_mat(velocities[i].data(), size);
            Vector      weight_mat(weight.data(), size);
            ConstVector grad_mat(grad.data(), size);

            velocity_mat.array() = gamma * velocity_mat.array() + m_lr * grad_mat.array();

            weight_mat.array() -= velocity_mat.array();
        }
    }
}

float Momentum::get_lr() const {
    return m_lr;
}

void Momentum::set_lr(float lr) {
    m_lr = lr;
}

} // namespace optimizer
} // namespace deepworks
