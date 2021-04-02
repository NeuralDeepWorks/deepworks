#pragma once

#include <vector>

#include <deepworks/tensor.hpp>

namespace deepworks {

struct IBackend {
    virtual void forward (const std::vector<Tensor>& inputs,
                                std::vector<Tensor>& outputs) = 0;
    virtual void backward(const std::vector<Tensor>& inputs,
                          const std::vector<Tensor>& outputs,
                          const std::vector<Tensor>& grad_outputs) = 0;
};

} // namespace deepworks
