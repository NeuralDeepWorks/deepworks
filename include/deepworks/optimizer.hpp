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

} // namespace loss
} // namespace deepworks
