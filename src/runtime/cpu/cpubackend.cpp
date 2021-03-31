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
    m_grad.resize(info.num_data_nodes);
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
                    m_grad[d.id].allocate(d.ph.shape());
                }

                // FIXME: In fact we don't need to allocate memory
                // to store gradients for inputs, because the aren't
                // used, but backend isn't ready to implement such logic.
                if (d.s == graph::Data::Storage::INPUT) {
                    m_grad[d.id].allocate(d.ph.shape());
                }

                break;
            }
            default:
                DeepWorks_Assert(false && "Unsupported node type");
        }
    }
}

void dwcpu::CPUBackend::bind(const std::vector<deepworks::Tensor>& tensors,
                             const std::vector<ade::NodeHandle>  & handles,
                                   std::vector<deepworks::Tensor>& mem) {
    for (int i = 0; i < handles.size(); ++i) {
        const auto& d = m_tgraph.metadata(handles[i]).get<graph::Data>();
        mem[d.id] = tensors[i];
    }
}

void dwcpu::CPUBackend::forward(const std::vector<deepworks::Tensor>& inputs,
                                      std::vector<deepworks::Tensor>& outputs) {
    const auto& info = m_tgraph.metadata().get<graph::Info>();
    bind(inputs , info.in_nhs,  m_mem);
    bind(outputs, info.out_nhs, m_mem);

    auto extract = [this](int id) { return m_mem[id]; };

    dwcpu::CPUGraph cpu_graph(m_graph);
    for (auto&& nh : m_ops) {
        std::vector<deepworks::Tensor> ins;
        std::vector<deepworks::Tensor> outs;

        // NB: Get layer ins/outs.
        const auto& op = m_tgraph.metadata(nh).get<graph::Op>();
        std::transform(op.in_ids.begin() , op.in_ids.end() , std::back_inserter(ins),  extract);
        std::transform(op.out_ids.begin(), op.out_ids.end(), std::back_inserter(outs), extract);

        // NB: Run layer.
        const auto& cpulayer = cpu_graph.metadata(nh).get<dwcpu::CPUImpl>().impl;
        cpulayer->forward(ins, outs);
    }
}

void dwcpu::CPUBackend::backward(const std::vector<deepworks::Tensor>& inputs,
                                 const std::vector<deepworks::Tensor>& outputs,
                                 const std::vector<deepworks::Tensor>& grad_outputs) {
    const auto& info = m_tgraph.metadata().get<graph::Info>();
    bind(inputs      , info.in_nhs , m_mem);
    bind(outputs     , info.out_nhs, m_mem);
    bind(grad_outputs, info.out_nhs, m_grad);

    auto extract_mem  = [this](int id) { return m_mem[id];  };
    auto extract_grad = [this](int id) { return m_grad[id]; };

    dwcpu::CPUGraph cpu_graph(m_graph);
    for (int i = m_ops.size() - 1; i >= 0; --i) {
        auto nh = m_ops[i];

        std::vector<deepworks::Tensor> inputs;
        std::vector<deepworks::Tensor> outputs;
        std::vector<deepworks::Tensor> grad_outputs;
        std::vector<deepworks::Tensor> grad_inputs;

        const auto& op = m_tgraph.metadata(nh).get<graph::Op>();
        std::transform(op.in_ids.begin() , op.in_ids.end() , std::back_inserter(grad_inputs) , extract_grad);

        // NB: Run layer.
        std::transform(op.in_ids.begin() , op.in_ids.end() , std::back_inserter(inputs)      , extract_mem);
        std::transform(op.out_ids.begin(), op.out_ids.end(), std::back_inserter(outputs)     , extract_mem);
        std::transform(op.out_ids.begin(), op.out_ids.end(), std::back_inserter(grad_outputs), extract_grad);
        const auto& cpulayer = cpu_graph.metadata(nh).get<dwcpu::CPUImpl>().impl;

        // FIXME: It isn't neccessary to compute gradients if parameters is freezed.
        cpulayer->updateGradients(inputs, grad_outputs);
        cpulayer->backward(inputs, outputs, grad_outputs, grad_inputs);
    }
}
