#include <deepworks/nn.hpp>
#include <deepworks/call.hpp>

deepworks::Linear::Linear(int units, std::string name)
    : BaseOp<deepworks::Linear>(deepworks::LayerInfo(name, "Linear")) {
    m_info.impl().attrs["units"] = units;
}

deepworks::Shape deepworks::Linear::output_shape(const deepworks::Shape& input) {
    return {input[0], m_info.impl().attrs["units"].get<int>()};
}

deepworks::ReLU::ReLU(std::string name)
    : BaseOp<deepworks::ReLU>(deepworks::LayerInfo(name, "ReLU")) {
}

deepworks::Shape deepworks::ReLU::output_shape(const deepworks::Shape& input) {
    return input;
}

deepworks::Softmax::Softmax(std::string name)
    : BaseOp<deepworks::Softmax>(LayerInfo(name, "Softmax")) {
}

deepworks::Shape deepworks::Softmax::output_shape(const deepworks::Shape& input) {
    return input;
}
