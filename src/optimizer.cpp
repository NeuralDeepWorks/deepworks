#include <Eigen/Core>

#include <deepworks/initializers.hpp>
#include <deepworks/optimizer.hpp>
#include <deepworks/utils/assert.hpp>

namespace deepworks {
namespace optimizer {

using ConstVector = Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;
using Vector = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;

SGD::SGD(ParamMap& params, float lr) : m_params(params), m_lr(lr) {}

void SGD::step() {
    for (auto& [name, param]: m_params) {
        if (param.is_trainable()) {
            auto       weight = param.data();
            const auto grad   = param.grad();

            DeepWorks_Assert(grad.shape() == weight.shape());

            const size_t size = grad.total();

            Vector      weight_mat(weight.data(), size);
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

Adam::Adam(ParamMap& params, float lr, std::array<float, 2> betas, float epsilon,
           size_t num_iterations)
        : m_params(params), m_lr(lr), m_betas(betas),
          m_epsilon(epsilon), m_num_iterations(num_iterations) {
    for (auto& [name, param]: m_params) {
        m_moving_mean.emplace    (name, Tensor::zeros(param.data().shape()));
        m_moving_variance.emplace(name, Tensor::zeros(param.data().shape()));
    }
}

void Adam::step() {
    m_num_iterations++;

    for (auto& [name, param] : m_params) {
        if (param.is_trainable()) {
            auto       weight = param.data();
            const auto grad   = param.grad();

            DeepWorks_Assert(grad.shape() == weight.shape());

            const size_t size = grad.total();

            Vector      moving_mean_mat    (m_moving_mean.at(name).data()    , size);
            Vector      moving_variance_mat(m_moving_variance.at(name).data(), size);
            Vector      weight_mat         (weight.data()                    , size);
            ConstVector grad_mat           (grad.data()                      , size);

            moving_mean_mat.array()     = m_betas[0] * moving_mean_mat.array()
                                          + (1 - m_betas[0]) * grad_mat.array();

            moving_variance_mat.array() = m_betas[1] * moving_variance_mat.array()
                                          + (1 - m_betas[1]) * grad_mat.cwiseAbs2().array();

            auto mean_hat     = moving_mean_mat.array()     / (1 - std::pow(m_betas[0], m_num_iterations));
            auto variance_hat = moving_variance_mat.array() / (1 - std::pow(m_betas[1], m_num_iterations));

            weight_mat.array() -= m_lr * mean_hat.array() / (variance_hat.cwiseSqrt().array() + m_epsilon).array();
        }
    }
}

float Adam::get_lr() const {
    return m_lr;
}

void Adam::set_lr(float lr) {
    m_lr = lr;
}

} // namespace optimizer
} // namespace deepworks
