#pragma once

#include <vector>

#include <deepworks/tensor.hpp>

namespace deepworks {

struct IBackend {
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& tensors)  = 0;
    virtual std::vector<Tensor> backward(const std::vector<Tensor>& tensors) = 0;
};

} // namespace deepworks
