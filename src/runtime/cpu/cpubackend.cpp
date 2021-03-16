#include "runtime/cpu/cpubackend.hpp"

#include "util/assert.hpp"

#include <ade/typed_graph.hpp>

namespace deepworks {
namespace cpu {

struct CPUImpl {
    static const char *name() { return "CPUImpl"; }
    std::shared_ptr<ICPULayer> impl;
};

using CPUGraph = ade::TypedGraph<CPUImpl>;
} // namespace cpu
} // namespace deepworks

namespace dwcpu = deepworks::cpu;
namespace graph  = deepworks::graph;

dwcpu::CPUBackend::CPUBackend(graph::Graph& graph)
    : m_graph(graph), m_tgraph(m_graph) {
    auto sorted = m_tgraph.metadata().get<ade::passes::TopologicalSortData>().nodes();
    const auto& info = m_tgraph.metadata().get<graph::Info>();

    m_mem.resize(info.num_data_nodes);

    dwcpu::CPUGraph cpu_graph(m_graph);
    for (auto&& nh : sorted) {
        switch (m_tgraph.metadata(nh).get<graph::Type>().t)
        {
            case graph::Type::OP: {
                // Instance layers.
                const auto& op = m_tgraph.metadata(nh).get<graph::Op>();
                cpu_graph.metadata(nh).set(dwcpu::CPUImpl{dwcpu::ICPULayer::create(op.info)});
                m_ops.push_back(nh);
                break;
            }
            case graph::Type::DATA: {
                const auto& d = m_tgraph.metadata(nh).get<graph::Data>();
                // NB: Allocate all internal tensors.
                if (d.s == graph::Data::Storage::INTERNAL) {
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

void dwcpu::CPUBackend::bind(const std::vector<deepworks::Tensor>& tensors,
                             const std::vector<ade::NodeHandle>  & handles) {
    for (int i = 0; i < handles.size(); ++i) {
        const auto& d = m_tgraph.metadata(handles[i]).get<graph::Data>();
        m_mem[d.id] = tensors[i];
    }
}

void dwcpu::CPUBackend::forward(const std::vector<deepworks::Tensor>& inputs,
                                      std::vector<deepworks::Tensor>& outputs) {
    const auto& info = m_tgraph.metadata().get<graph::Info>();
    bind(inputs , info.in_nhs);
    bind(outputs, info.out_nhs);

    auto extract = [this](int id) { return m_mem[id]; };

    dwcpu::CPUGraph cpu_graph(m_graph);
    for (auto&& nh : m_ops) {
        std::vector<deepworks::Tensor> inputs;
        std::vector<deepworks::Tensor> outputs;

        // NB: Get layer inputs/outputs.
        const auto& op = m_tgraph.metadata(nh).get<graph::Op>();
        std::transform(op.in_ids.begin() , op.in_ids.end() , std::back_inserter(inputs),  extract);
        std::transform(op.out_ids.begin(), op.out_ids.end(), std::back_inserter(outputs), extract);

        // NB: Run layer.
        const auto& cpulayer = cpu_graph.metadata(nh).get<dwcpu::CPUImpl>().impl;
        cpulayer->forward(inputs, outputs);
    }
}

void dwcpu::CPUBackend::backward(const std::vector<deepworks::Tensor>& inputs,
                                       std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(false && "Not implemented");
}
