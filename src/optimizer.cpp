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

SGDMomentum::SGDMomentum(Parameters& params, float lr, float gamma) : m_params(params), m_lr(lr), gamma(gamma) {
    for (auto& param: m_params) {
        velocities.emplace_back(Tensor{param.data().shape()});
        initializer::zeros(velocities.back());
    }
}

void SGDMomentum::step() {
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

float SGDMomentum::get_lr() const {
    return m_lr;
}

void SGDMomentum::set_lr(float lr) {
    m_lr = lr;
}

Adam::Adam(Parameters& params, float lr, float beta_one, float beta_second, float epsilon,
           size_t num_iterations)
        : m_params(params), m_lr(lr), beta_one(beta_one),
          beta_second(beta_second), epsilon(epsilon), num_iterations(num_iterations) {
    for (auto& param: m_params) {
        moving_mean.emplace_back(Tensor{param.data().shape()});
        moving_variance.emplace_back(Tensor{param.data().shape()});

        initializer::zeros(moving_mean.back());
        initializer::zeros(moving_variance.back());
    }
}

void Adam::step() {
    num_iterations++;

    for (size_t i = 0; i < m_params.size(); ++i) {
        if (m_params[i].is_trainable()) {
            auto       weight = m_params[i].data();
            const auto grad   = m_params[i].grad();

            DeepWorks_Assert(grad.shape() == weight.shape());

            const size_t size = grad.total();

            Vector      moving_mean_mat(moving_mean[i].data(), size);
            Vector      moving_variance_mat(moving_variance[i].data(), size);
            Vector      weight_mat(weight.data(), size);
            ConstVector grad_mat(grad.data(), size);

            moving_mean_mat.array() = beta_one * moving_mean_mat.array() + (1 - beta_one) * grad_mat.array();
            moving_variance_mat.array() = beta_second * moving_variance_mat.array()
                                          + (1 - beta_second) * grad_mat.cwiseAbs2().array();

            auto mean_hat     = moving_mean_mat.array() / (1 - std::pow(beta_one, num_iterations));
            auto variance_hat = moving_variance_mat.array() / (1 - std::pow(beta_second, num_iterations));

            weight_mat.array() -= m_lr * mean_hat.array() / (variance_hat.cwiseSqrt().array() + epsilon).array();
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
