#pragma once

#include <deepworks/placeholder.hpp>
#include <deepworks/layer_info.hpp>
#include <deepworks/call.hpp>

namespace deepworks {

template <typename D>
struct BaseOp {
    BaseOp(LayerInfo info) : m_info(std::move(info)) { }
    Placeholder operator()(Placeholder in) {
        Call call{m_info};
        call.pass({in});
        return call.create(static_cast<D*>(this)->output_shape(in.shape()));
    }

    LayerInfo m_info;
};

struct Linear : BaseOp<Linear> {
    Linear(int units, std::string name);
    Shape output_shape(const Shape& input);
};

struct ReLU : BaseOp<ReLU> {
    ReLU(std::string name);
    Shape output_shape(const Shape& input);
};

struct Softmax : BaseOp<Softmax> {
    Softmax(std::string name);
    Shape output_shape(const Shape& input);
};

} // namespace deepworks
