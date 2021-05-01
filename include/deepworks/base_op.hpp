#pragma once

#include <deepworks/placeholder.hpp>
#include <deepworks/call.hpp>
#include <deepworks/layer_info.hpp>

namespace deepworks {

template <typename R, class C, typename ... Types>
constexpr size_t get_num_ins( R(C::*)(Types ...)) {
    return sizeof...(Types);
}

template <typename D>
struct BaseOp {
    // NB: Get number of inputs arguments for operator()
    // it must be the same as for outShape
    static constexpr size_t num_in = get_num_ins(&D::outShape);

    BaseOp(LayerInfo&& info) : m_info(std::move(info)) { }

    template <typename... Shapes>
    void init(const Shapes&... in_shapes) { /* do nothing */ }

    template <typename... Shapes>
    Shape outShape(const Shapes&... in_shapes) {
        throw std::logic_error("outShape for operation isn't implemented");
    };

    // NB: User friendly operator()
    template <typename... Placeholders>
    Placeholder operator()(Placeholders... ins) {
        auto get_shape = [](Placeholder ph) { return ph.shape(); };
        static_cast<D*>(this)->init(get_shape(ins)...);

        Call call{m_info};
        call.pass({ins...});
        return call.create(static_cast<D*>(this)->outShape(get_shape(ins)...));
    }

    // NB: Generic operator() which works with vector
    Placeholder operator()(const Placeholders& phs) {
        return call_impl(phs, std::make_index_sequence<num_in>()); 
    }

    template<size_t... I>
    Placeholder call_impl(const Placeholders& phs, std::index_sequence<I...>) {
        auto get_ph = [](const Placeholders& phs, int idx) { return phs[idx]; };
        return (*this)(get_ph(phs, I)...);
    }

    LayerInfo m_info;
};

} // namespace deepworks
