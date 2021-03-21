#pragma once

#include <deepworks/tensor.hpp>

namespace deepworks {

class Parameter {
public:
    Parameter(Tensor&& data, bool is_trainable=true);

    const Tensor& data() const;
          Tensor& data();

    const Tensor& grad() const;
          Tensor& grad();

    void train(bool mode);
    bool is_trainable() const;

private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

using Parameters = std::vector<Parameter>;

} // namespace deepworks
