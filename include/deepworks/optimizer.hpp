#pragma once

#include <deepworks/tensor.hpp>
#include <deepworks/parameter.hpp>

namespace deepworks {
namespace optimizer {

class SGD {
public:

    explicit SGD(Parameters& params, float lr);

    void step();

    float get_learning_rate();

    void set_learning_rate(float lr);

private:
    float learning_rate;
    Parameters& parameters;
};

} // namespace loss
} // namespace deepworks
