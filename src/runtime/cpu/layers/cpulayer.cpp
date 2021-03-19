#include "runtime/cpu/layers/cpulayer.hpp"

#include "runtime/cpu/layers/cpurelu.hpp"
#include "runtime/cpu/layers/cpusoftmax.hpp"


deepworks::cpu::ICPULayer::Ptr deepworks::cpu::ICPULayer::create(deepworks::LayerInfo info) {
    // FIXME: Should be the map[string]ptr
    if (info.type() == "ReLU") {
        return std::make_shared<deepworks::cpu::CPUReLU>(info);
    }

    if (info.type() == "Softmax") {
        return std::make_shared<deepworks::cpu::CPUSoftmax>(info);
    }

    DeepWorks_Assert(false && "Unsupported layer type in CPUBackend");
    return nullptr;
}

deepworks::cpu::ICPULayer::ICPULayer(deepworks::LayerInfo info)
    : m_info(std::move(info)) {
}
