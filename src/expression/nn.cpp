#include <numeric>
#include <functional>

#include <deepworks/nn.hpp>
#include <deepworks/call.hpp>
#include <deepworks/tensor.hpp>
#include <deepworks/initializers.hpp>

#include "util/assert.hpp"

enum Input  {N, C, H, W};
enum Kernel {KH, KW};

namespace dw = deepworks;

dw::Linear::Linear(int units, std::string name)
    : BaseOp<dw::Linear>(dw::LayerInfo(std::move(name), "Linear")) {
    m_info.impl().attrs["units"] = units;
}

template<class T>
struct get_in {
    static T get(const dw::Attributes&           attrs,
                 const std::vector<std::string>& order,
                 int                             idx) {
        return attrs.at(order[idx]).get<T>();
    }
};

template <typename T, typename... Types>
struct make_typed_layer {
    static constexpr int num_in = sizeof...(Types);

    make_typed_layer(const std::vector<std::string>& order = {})
        : m_order(order) {
    }

    T operator()(const dw::Attributes& attrs, const std::string& name) {
        return call(attrs, m_order, name, std::make_index_sequence<num_in>());
    }

    template<size_t... I>
    T call(const dw::Attributes&           attrs,
           const std::vector<std::string>& order,
           const std::string&              name,
           std::index_sequence<I...>) {
        return T(get_in<Types>::get(attrs, order, I)..., name);
    }

    std::vector<std::string> m_order;
};

dw::Placeholder dw::make_layer(const std::string     & type,
                               const std::string     & name,
                               const dw::Attributes  & attrs,
                               const dw::Placeholders& inputs) {
    using make_layer_f = std::function<dw::Placeholder()>;
    using table_t      = std::unordered_map<std::string, make_layer_f>;

    auto make_linear = [&]{
        return make_typed_layer<dw::Linear, int>({"units"})(attrs, name)(inputs);
    };

    auto make_relu = [&]{
        return make_typed_layer<dw::ReLU>()(attrs, name)(inputs);
    };

    auto make_elu = [&]{
        return make_typed_layer<dw::ELU, float>({"alpha"})(attrs, name)(inputs);
    };

    auto make_batchnorm1d = [&]{
        return make_typed_layer<dw::BatchNorm1D, float, float>
            ({"eps", "alpha"})(attrs, name)(inputs);
    };

    auto make_softmax = [&]{
        return make_typed_layer<dw::Softmax>()(attrs, name)(inputs);
    };

    auto make_sigmoid = [&]{
        return make_typed_layer<dw::Sigmoid>()(attrs, name)(inputs);
    };

    auto make_maxpooling = [&]{
        using array2_t = std::array<int, 2>;
        return make_typed_layer<dw::MaxPooling, array2_t, array2_t, array2_t>
            ({"kernel", "padding", "stride"})(attrs, name)(inputs);
    };

    auto make_convolution = [&]{
        using array2_t = std::array<int, 2>;
        return make_typed_layer<dw::Convolution, int, array2_t, array2_t, array2_t>
            ({"out_channels", "kernel", "padding", "stride"})(attrs, name)(inputs);
    };

    auto make_leakyrelu = [&]{
        return make_typed_layer<dw::LeakyReLU, float>
            ({"alpha"})(attrs, name)(inputs);
    };

    auto make_dropout = [&]{
        return make_typed_layer<dw::Dropout, float>({"p"})(attrs, name)(inputs);
    };

    table_t supported_layers = {
        {"Linear"     , make_linear},
        {"ReLU"       , make_relu},
        {"BatchNorm1D", make_batchnorm1d},
        {"Softmax"    , make_softmax},
        {"MaxPooling" , make_maxpooling},
        {"MaxPooling" , make_convolution},
        {"ELU"        , make_elu},
        {"LeakyReLU"  , make_leakyrelu},
        {"Sigmoid"    , make_sigmoid},
        {"Dropout"    , make_dropout},
    };

    auto f_it = supported_layers.find(type);
    if (f_it != supported_layers.end()) {
        return f_it->second();
    }

    DeepWorks_Throw() << "Can't create " << type << " layer";
}

void dw::Linear::init(const Shape& in_shape) {
    int units = m_info.impl().attrs["units"].get<int>();

    auto second_shape = std::accumulate(in_shape.begin() + 1,
            in_shape.end(), 1, std::multiplies<int>());
    // NB: Init weight.
    m_info.impl().params.emplace("weight",
            dw::Tensor::xavierUniform({units, second_shape}));

    // NB: Init bias.
    m_info.impl().params.emplace("bias", dw::Tensor::zeros({units}));
}

dw::Shape dw::Linear::outShape(const dw::Shape& in_shape) {
    DeepWorks_Assert(in_shape.size() != 1u && "Linear layer doesn't work with 1D tensors");
    int units = m_info.impl().attrs["units"].get<int>();
    return {in_shape[0], units};
}

dw::ReLU::ReLU(std::string name)
    : BaseOp<dw::ReLU>(dw::LayerInfo(std::move(name), "ReLU")) {
}

dw::Shape dw::ReLU::outShape(const dw::Shape& in_shape) {
    return in_shape;
}

dw::Softmax::Softmax(std::string name)
    : BaseOp<dw::Softmax>(LayerInfo(std::move(name), "Softmax")) {
}

dw::Shape dw::Softmax::outShape(const dw::Shape& in_shape) {
    return in_shape;
}

dw::BatchNorm1D::BatchNorm1D(float eps, float alpha, std::string name)
    : BaseOp<dw::BatchNorm1D>(LayerInfo(std::move(name), "BatchNorm1D")) {
    m_info.impl().attrs["eps"] = eps;
    m_info.impl().attrs["alpha"] = alpha;
}

void dw::BatchNorm1D::init(const Shape& in_shape) {
    // NB: Init trainable parameters and buffers.
    m_info.impl().params.emplace ("gamma"       , dw::Tensor::constant({in_shape[1]}, 1.0));
    m_info.impl().params.emplace ("beta"        , dw::Tensor::zeros({in_shape[1]}));
    m_info.impl().buffers.emplace("running_mean", dw::Tensor::zeros({in_shape[1]}));
    m_info.impl().buffers.emplace("running_var" , dw::Tensor::zeros({in_shape[1]}));
}

dw::Shape dw::BatchNorm1D::outShape(const dw::Shape& in_shape) {
    return in_shape;
}

dw::ELU::ELU(float alpha, std::string name)
    : BaseOp<dw::ELU>(dw::LayerInfo(std::move(name), "ELU")) {
    m_info.impl().attrs["alpha"] = alpha;
}

dw::Shape dw::ELU::outShape(const dw::Shape& in_shape) {
    return in_shape;
}

dw::MaxPooling::MaxPooling(const std::array<int, 2>& kernel,
                                  const std::array<int, 2>& padding,
                                  const std::array<int, 2>& stride,
                                  std::string name)
    : BaseOp<dw::MaxPooling>(LayerInfo(std::move(name), "MaxPooling")) {
    m_info.impl().attrs["kernel"] = kernel;
    m_info.impl().attrs["padding"] = padding;
    m_info.impl().attrs["stride"] = stride;
}

dw::Shape dw::MaxPooling::outShape(const dw::Shape& in_shape) {
    DeepWorks_Assert(in_shape.size() == 4u && "MaxPooling layer works only with 4D tensors");
    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    int h_out = (in_shape[Input::H] + 2 * padding[Kernel::KH] - kernel[Kernel::KH]) / stride[Kernel::KH] + 1;
    int w_out = (in_shape[Input::W] + 2 * padding[Kernel::KW] - kernel[Kernel::KW]) / stride[Kernel::KW] + 1;
    return {in_shape[0], in_shape[1], h_out, w_out};
}

dw::Convolution::Convolution(int out_channels,
                                    const std::array<int, 2>& kernel,
                                    const std::array<int, 2>& padding,
                                    const std::array<int, 2>& stride,
                                    std::string name)
    : BaseOp<dw::Convolution>(LayerInfo(std::move(name), "Convolution")) {
    m_info.impl().attrs["out_channels"] = out_channels;
    m_info.impl().attrs["kernel"] = kernel;
    m_info.impl().attrs["padding"] = padding;
    m_info.impl().attrs["stride"] = stride;
}

void dw::Convolution::init(const Shape& in_shape) {
    int out_channels = m_info.impl().attrs["out_channels"].get<int>();
    auto kernel = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();

    // NB: Init weight.
    m_info.impl().params.emplace("weight", dw::Tensor::xavierUniform({out_channels,
                                                                      in_shape[Input::C],
                                                                      kernel[Kernel::KH],
                                                                      kernel[Kernel::KW]}));
    // NB: Init bias.
    m_info.impl().params.emplace("bias", dw::Tensor::zeros({out_channels}));
}

dw::Shape dw::Convolution::outShape(const dw::Shape& in_shape) {
    DeepWorks_Assert(in_shape.size() == 4u && "Convolution layer works only with 4D tensors");
    int out_channels = m_info.impl().attrs["out_channels"].get<int>();
    auto kernel  = m_info.impl().attrs["kernel"].get<std::array<int, 2>>();
    auto padding = m_info.impl().attrs["padding"].get<std::array<int, 2>>();
    auto stride  = m_info.impl().attrs["stride"].get<std::array<int, 2>>();

    int h_out = (in_shape[Input::H] + 2 * padding[Kernel::KH] - kernel[Kernel::KH]) / stride[Kernel::KH] + 1;
    int w_out = (in_shape[Input::W] + 2 * padding[Kernel::KW] - kernel[Kernel::KW]) / stride[Kernel::KW] + 1;
    return {in_shape[0], out_channels, h_out, w_out};
}

dw::LeakyReLU::LeakyReLU(float alpha, std::string name)
    : BaseOp<dw::LeakyReLU>(dw::LayerInfo(std::move(name), "LeakyReLU")) {
    m_info.impl().attrs["alpha"] = alpha;
}

dw::Shape dw::LeakyReLU::outShape(const dw::Shape& in_shape) {
    return in_shape;
}

deepworks::Sigmoid::Sigmoid(std::string name)
        : BaseOp<deepworks::Sigmoid>(deepworks::LayerInfo(std::move(name), "Sigmoid")) {
}

deepworks::Shape deepworks::Sigmoid::outShape(const deepworks::Shape& in_shape) {
    return in_shape;
}

deepworks::Dropout::Dropout(float p, std::string name)
    : BaseOp<deepworks::Dropout>(deepworks::LayerInfo(std::move(name), "Dropout")) {
    m_info.impl().attrs["p"] = p;
}

void dw::Dropout::init(const Shape& in_shape) {
    m_info.impl().params.emplace("mask", dw::Tensor::zeros(in_shape));
}

deepworks::Shape deepworks::Dropout::outShape(const deepworks::Shape& in_shape) {
    return in_shape;
}
