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
    float m_lr;
    ParamMap& m_params;
};

class SGDMomentum {
public:
    explicit SGDMomentum(ParamMap& params, float lr, float gamma = 0.9f);

    void step();

    float get_lr() const;

    void set_lr(float lr);

private:
    float m_lr;
    float gamma;
    ParamMap& m_params;
    std::vector<Tensor> velocities;
};

} // namespace loss
} // namespace deepworks
