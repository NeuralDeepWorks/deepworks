#include <deepworks/layers.hpp>

#include "call.hpp"

//////////////////////// Linear //////////////////////////
deepworks::Linear::Linear(int units, std::string name)
    : m_info(name, "Linear") {
        m_info.priv().attrs["units"] = units;
    }

deepworks::Shape deepworks::Linear::output_shape(const deepworks::Shape& input) {
    return {input[0], m_info.priv().attrs["units"].get<int>()};
}

// FIXME: How to hide this ugly repeated code ?
deepworks::Placeholder deepworks::Linear::operator()(deepworks::Placeholder in) {
    deepworks::Call call{m_info};
    call.pass({in});
    return call.create(output_shape(in.shape()));
}

//////////////////////// Linear //////////////////////////

//////////////////////// ReLU //////////////////////////
deepworks::ReLU::ReLU(std::string name)
    : m_info(name, "ReLU") {
    }

deepworks::Shape deepworks::ReLU::output_shape(const deepworks::Shape& input) {
    return input;
}

// FIXME: How to hide this ugly repeated code ?
deepworks::Placeholder deepworks::ReLU::operator()(deepworks::Placeholder in) {
    deepworks::Call call{m_info};
    call.pass({in});
    return call.create(output_shape(in.shape()));
}

//////////////////////// ReLU //////////////////////////

//////////////////////// Softmax //////////////////////////
deepworks::Softmax::Softmax(std::string name)
    : m_info(name, "Softmax") {
    }

deepworks::Shape deepworks::Softmax::output_shape(const deepworks::Shape& input) {
    return input;
}

// FIXME: How to hide this ugly repeated code ?
deepworks::Placeholder deepworks::Softmax::operator()(deepworks::Placeholder in) {
    deepworks::Call call{m_info};
    call.pass({in});
    return call.create(output_shape(in.shape()));
}

//////////////////////// Softmax //////////////////////////
