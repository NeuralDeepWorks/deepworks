#pragma once

#include <deepworks/tensor.hpp>
#include <deepworks/parameter.hpp>

namespace deepworks {
namespace optimizer {

class SGD {
public:
    explicit SGD(ParamMap& params, float lr);

    void step();

    float get_lr() const;

    void set_lr(float lr);

private:
    float     m_lr;
    ParamMap& m_params;
};

class SGDMomentum {
public:
    explicit SGDMomentum(ParamMap& params, float lr, float gamma = 0.9f);

    void step();

    float get_lr() const;

    void set_lr(float lr);

private:
    float     m_lr;
    float     m_gamma;
    ParamMap& m_params;
    TensorMap m_velocities;
};


class Adam {
public:
    explicit Adam(ParamMap& params, float lr, std::array<float, 2> betas = {0.9f, 0.999f},
                  float epsilon = 0.001, size_t num_iterations = 0);

    void step();

    float get_lr() const;

    void set_lr(float lr);

private:
    float                m_lr;
    std::array<float, 2> m_betas;

    float     m_epsilon;
    size_t    m_num_iterations;

    ParamMap& m_params;
    TensorMap m_moving_mean;
    TensorMap m_moving_variance;
};

} // namespace loss
} // namespace deepworks
