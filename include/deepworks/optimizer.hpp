#pragma once

#include <deepworks/tensor.hpp>
#include <deepworks/parameter.hpp>

namespace deepworks {
namespace optimizer {

class SGD {
public:
    explicit SGD(Parameters& params, float lr);

    void step();

    float get_lr() const;

    void set_lr(float lr);

private:
    float m_lr;

    Parameters& m_params;
};

class SGDMomentum {
public:
    explicit SGDMomentum(Parameters& params, float lr, float gamma = 0.9f);

    void step();

    float get_lr() const;

    void set_lr(float lr);

private:
    float m_lr;
    float gamma;

    Parameters& m_params;
    std::vector<Tensor> velocities;
};


class Adam {
public:
    explicit Adam(Parameters& params, float lr, float beta_one = 0.9f,
                  float beta_second = 0.999f, float epsilon = 0.001, size_t num_iterations = 0);

    void step();

    float get_lr() const;

    void set_lr(float lr);

private:
    float m_lr;
    float beta_one;
    float beta_second;

    float epsilon;
    size_t num_iterations;

    Parameters& m_params;
    std::vector<Tensor> moving_mean;
    std::vector<Tensor> moving_variance;
};

} // namespace loss
} // namespace deepworks
