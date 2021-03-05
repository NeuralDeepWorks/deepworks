#include "runtime/cpu/cpubackend.hpp"
#include <iostream>

#include "util/assert.hpp"

std::vector<deepworks::Tensor>
deepworks::CPUBackend::forward(const std::vector<deepworks::Tensor>& tensors) {
    DeepWorks_Assert(tensors.size() == 1u && "Only single input forward is supported");
    auto in_tensor = tensors.front();

    auto& g_info = m_tg.metadata().get<deepworks::graph::GraphInfo>();

    DeepWorks_Assert(g_info.in_nhs.size() == 1u && "Only single input forward is supported");
    auto in_nh = g_info.in_nhs.front();
    auto& in_d = m_tg.metadata(in_nh).get<deepworks::graph::Data>();
    // NB: Bind inputs.
    m_mem[in_d.id] = in_tensor;

    // Create output tensor.
    std::vector<Tensor> outputs(g_info.out_nhs.size());
    auto out_nh = g_info.out_nhs.front();
    auto& out_d = m_tg.metadata(out_nh).get<deepworks::graph::Data>();
    outputs[0].allocate(out_d.ph.shape());

    // Inference Loop.
    DeepWorks_Assert(m_layers.size() == 1u && "Only single layer execution is supported now");
    auto& layer = m_layers.front();

    std::vector<Tensor> out_mem(1);
    out_mem[0] = m_mem[out_d.id];
    layer->forward({in_tensor}, out_mem);

    // Write back (deepcopy).
    m_mem[out_d.id].copyTo(outputs[0]);

    return outputs;
}

std::vector<deepworks::Tensor>
deepworks::CPUBackend::backward(const std::vector<deepworks::Tensor>& tensors) {
    DeepWorks_Assert(false && "Not implemented");
    return {};
}

deepworks::CPUBackend::CPUBackend(deepworks::graph::Graph& g,
                                  deepworks::graph::TypedGraph& tg,
                                  int batch_size)
    : m_g(g), m_tg(tg), m_bs(batch_size) {
    auto sorted = m_tg.metadata().get<ade::passes::TopologicalSortData>().nodes();
    const auto& g_info = m_tg.metadata().get<deepworks::graph::GraphInfo>();

    m_mem.resize(g_info.num_data_nodes);

    for (auto&& nh : sorted) {
        switch (m_tg.metadata(nh).get<deepworks::graph::Type>().t)
        {
            case deepworks::graph::Type::OP: {
                // Instance layers.
                const auto& op = m_tg.metadata(nh).get<deepworks::graph::Op>();
                m_layers.push_back(deepworks::cpu::ICPULayer::create(op.info));
                break;
            }
            case deepworks::graph::Type::DATA: {
                auto& d = m_tg.metadata(nh).get<deepworks::graph::Data>();
                // NB: Allocate all tensors except input.
                if (d.s != deepworks::graph::Data::Storage::INPUT) {
                    // FIXME: Shape should be already propagated.
                    // Stop using placeholder here, it left on expression step.
                    m_mem[d.id].allocate(d.ph.shape());
                }
                break;
            }
            default:
                DeepWorks_Assert(false && "Unsupported node type");
        }
    }
}
