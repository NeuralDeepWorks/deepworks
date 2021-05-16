#include "runtime/cpu/layers/cpulayer.hpp"

#include "runtime/cpu/layers/cpurelu.hpp"
#include "runtime/cpu/layers/cpulinear.hpp"
#include "runtime/cpu/layers/cpusoftmax.hpp"
#include "runtime/cpu/layers/cpubatchnorm1d.hpp"
#include "runtime/cpu/layers/cpubatchnorm2d.hpp"
#include "runtime/cpu/layers/cpuelu.hpp"
#include "runtime/cpu/layers/cpuleakyrelu.hpp"
#include "runtime/cpu/layers/cpumaxpooling.hpp"
#include "runtime/cpu/layers/cpuconvolution.hpp"
#include "runtime/cpu/layers/cpusigmoid.hpp"
#include "runtime/cpu/layers/cpudropout.hpp"
#include "runtime/cpu/layers/cpuglobalavgpooling.hpp"
#include "runtime/cpu/layers/cpuadd.hpp"

deepworks::cpu::ICPULayer::Ptr deepworks::cpu::ICPULayer::create(deepworks::LayerInfo info) {
    // FIXME: Should be the map[string]ptr
    if (info.type() == "ReLU") {
        return std::make_shared<deepworks::cpu::CPUReLU>(std::move(info));
    } else if (info.type() == "Linear") {
        return std::make_shared<deepworks::cpu::CPULinear>(std::move(info));
    } else if (info.type() == "Softmax") {
        return std::make_shared<deepworks::cpu::CPUSoftmax>(std::move(info));
    } else if (info.type() == "BatchNorm1D") {
        return std::make_shared<deepworks::cpu::CPUBatchNorm1D>(std::move(info));
    } else if (info.type() == "BatchNorm2D") {
        return std::make_shared<deepworks::cpu::CPUBatchNorm2D>(std::move(info));
    } else if (info.type() == "ELU") {
        return std::make_shared<deepworks::cpu::CPUELU>(std::move(info));
    } else if (info.type() == "LeakyReLU") {
        return std::make_shared<deepworks::cpu::LeakyReLU>(std::move(info));
    } else if (info.type() == "MaxPooling") {
        return std::make_shared<deepworks::cpu::CPUMaxPooling>(std::move(info));
    } else if (info.type() == "Convolution") {
        return std::make_shared<deepworks::cpu::CPUConvolution>(std::move(info));
    } else if (info.type() == "Sigmoid") {
        return std::make_shared<deepworks::cpu::Sigmoid>(std::move(info));
    } else if (info.type() == "Dropout") {
        return std::make_shared<deepworks::cpu::CPUDropout>(std::move(info));
    } else if (info.type() == "GlobalAvgPooling") {
        return std::make_shared<deepworks::cpu::CPUGlobalAvgPooling>(std::move(info));
    } else if (info.type() == "Add") {
        return std::make_shared<deepworks::cpu::CPUAdd>(std::move(info));
    }

    DeepWorks_Assert(false && "Unsupported layer type in CPUBackend");
    return nullptr;
}

deepworks::cpu::ICPULayer::ICPULayer(deepworks::LayerInfo&& info)
    : m_info(std::move(info)) {
}

void deepworks::cpu::ICPULayer::updateGradients(const std::vector<Tensor>& inputs,
                                                const std::vector<Tensor>& grad_outputs) {
}
