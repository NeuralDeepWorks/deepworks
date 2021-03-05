#pragma once

#include <memory>

#include <deepworks/placeholder.hpp>
#include <deepworks/layer_info.hpp>

namespace deepworks {

struct LayerInfo;

// FIXME: Avoid code duplication
class Linear {
public:
    Linear(int units, std::string name);
    Placeholder operator()(Placeholder in);
private:
    Shape output_shape(const Shape& input);
    LayerInfo m_info;
};

class ReLU {
public:
    ReLU(std::string name);
    Placeholder operator()(Placeholder in);
private:
    Shape output_shape(const Shape& input);
    LayerInfo m_info;
};

class Softmax {
public:
    Softmax(std::string name);
    Placeholder operator()(Placeholder in);
private:
    Shape output_shape(const Shape& input);
    LayerInfo m_info;
};

} // namespace deepworks
