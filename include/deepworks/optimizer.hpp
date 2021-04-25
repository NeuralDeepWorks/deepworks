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

} // namespace loss
} // namespace deepworks
