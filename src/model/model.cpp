#include <stack>
#include <unordered_set>

#include <deepworks/model.hpp>

#include "expression/call_impl.hpp"
#include "expression/placeholder_impl.hpp"
#include "model/model_impl.hpp"
#include "model/graphbuilder.hpp"
#include "runtime/cpu/cpubackend.hpp"

#include "util/assert.hpp"
#include <iostream>

using namespace deepworks::graph;

// FIXME: This function will be implemented as a graph pass a bit later.
// Let it be a static function.
static void initNodeId(TypedGraph& tgraph) {
    // NB: Should be graph pass as well.
    int id = 0;
    // NB: Setup data id.
    auto sorted = tgraph.metadata().get<ade::passes::TopologicalSortData>().nodes();
    for (auto&& nh : sorted) {
        if (tgraph.metadata(nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::DATA) {
            tgraph.metadata(nh).get<deepworks::graph::Data>().id = id;
            ++id;
        }
    }

    // NB: Setup op in/out ids.
    for (auto&& nh : sorted) {
        if (tgraph.metadata(nh).get<deepworks::graph::Type>().t != deepworks::graph::Type::OP) {
            continue;
        }
        DeepWorks_Assert(tgraph.metadata(nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::OP);
        auto& op = tgraph.metadata(nh).get<deepworks::graph::Op>();

        op.in_ids.reserve(nh->inNodes().size());
        for (auto&& in_nh : nh->inNodes()) {
            DeepWorks_Assert(tgraph.metadata(in_nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::DATA);
            op.in_ids.push_back(tgraph.metadata(in_nh).get<deepworks::graph::Data>().id);
        }

        op.out_ids.reserve(nh->outNodes().size());
        for (auto&& out_nh : nh->outNodes()) {
            DeepWorks_Assert(tgraph.metadata(out_nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::DATA);
            op.out_ids.push_back(tgraph.metadata(out_nh).get<deepworks::graph::Data>().id);
        }
    }
}

deepworks::Model::Model(deepworks::Placeholder in, deepworks::Placeholder out)
    : deepworks::Model(deepworks::Placeholders{std::move(in)},
                       deepworks::Placeholders{std::move(out)}) {
}

deepworks::Model::Model(deepworks::Placeholders ins, deepworks::Placeholders outs)
    : m_impl(new Impl(std::move(ins), std::move(outs))) {
}

const deepworks::Placeholders& deepworks::Model::inputs() const {
    return m_impl->m_inputs;
}

const deepworks::Placeholders& deepworks::Model::outputs() const {
    return m_impl->m_outputs;
}

const deepworks::Layers& deepworks::Model::layers() const {
    return m_impl->m_layers;
}

deepworks::Layers& deepworks::Model::layers() {
    return m_impl->m_layers;
}

deepworks::Layer deepworks::Model::getLayer(const std::string& name) {
    auto it = m_impl->m_layers_map.find(name);
    DeepWorks_Assert(it != m_impl->m_layers_map.end() && "Layer with that name not found");
    return it->second;
}

deepworks::ParamMap& deepworks::Model::params() {
    return m_impl->m_params;
}

void deepworks::Model::train(bool mode) {
    for (auto& [name, parameter] : params()) {
        parameter.train(mode);
    }
}

deepworks::Model::Impl::Impl(deepworks::Placeholders ins,
                             deepworks::Placeholders outs)
    : m_tgraph(m_graph), m_inputs(std::move(ins)), m_outputs(std::move(outs)) {
        deepworks::GraphBuilder builder(m_graph);
        builder.build(m_inputs, m_outputs);

        // NB: Sort nodes in graph.
        ade::passes::PassContext context{m_graph};
        ade::passes::TopologicalSort()(context);
        auto sorted = m_tgraph.metadata().get<ade::passes::TopologicalSortData>().nodes();

        // NB: Create layers for user.
        // It also can a graph pass, but let's keep it separately so far.
        for (auto&& nh : sorted) {
            if (m_tgraph.metadata(nh).get<Type>().t == Type::OP) {
                // NB: Collect layer inputs.
                Placeholders inputs;
                inputs.reserve(nh->inNodes().size());
                for (auto&& in_nh : nh->inNodes()) {
                    DeepWorks_Assert(m_tgraph.metadata(in_nh).get<Type>().t == Type::DATA);
                    inputs.push_back(m_tgraph.metadata(in_nh).get<Data>().ph);
                }

                // NB: Collect layer outputs.
                Placeholders outputs;
                outputs.reserve(nh->outNodes().size());
                for (auto&& out_nh : nh->outNodes()) {
                    DeepWorks_Assert(m_tgraph.metadata(out_nh).get<Type>().t == Type::DATA);
                    outputs.push_back(m_tgraph.metadata(out_nh).get<Data>().ph);
                }

                auto&& info = m_tgraph.metadata(nh).get<Op>().info;
                // FIXME: Check that emplace performed without errors. (I'm so lazy)
                auto it = m_layers_map.emplace(info.name(),
                        Layer{info, std::move(inputs), std::move(outputs)}).first;
                m_layers.emplace_back(it->second);

                // NB: Collect all parameters from every layer. (Used by optimizer)
                for (auto&& [name, p] : it->second.params()) {
                    auto it = m_params.emplace(info.name() + std::string(".") + name, p).first;
                    m_state.emplace(it->first, p.data());
                }

                // NB: Push named buffers to state.
                for (auto&& [name, b] : it->second.buffers()) {
                    m_state.emplace(info.name() + std::string(".") + name, b);
                }
            }
        }

        initNodeId(m_tgraph);
}

const deepworks::Model::StateDict& deepworks::Model::state() const {
    return m_impl->m_state;
}

deepworks::Model::StateDict& deepworks::Model::state() {
    return m_impl->m_state;
}

void deepworks::Model::compile() {
    m_impl->m_backend = std::make_shared<deepworks::cpu::CPUBackend>(m_impl->m_graph);
}

void deepworks::Model::forward (const deepworks::Tensor& input,
                                      deepworks::Tensor& output) {
    std::vector<Tensor> outputs{output};
    forward({input}, outputs);
}

void deepworks::Model::backward(const deepworks::Tensor& input,
                                const deepworks::Tensor& output,
                                const deepworks::Tensor& grad_output) {
    backward(std::vector<deepworks::Tensor>{input},
             std::vector<deepworks::Tensor>{output},
             std::vector<deepworks::Tensor>{grad_output});
}

void deepworks::Model::forward(const std::vector<deepworks::Tensor>& inputs,
                                     std::vector<deepworks::Tensor>& outputs) {
    DeepWorks_Assert(m_impl->m_backend && "Model wasn't compiled !")
    m_impl->m_backend->forward(inputs, outputs);
}

void deepworks::Model::backward(const std::vector<deepworks::Tensor>& inputs,
                                const std::vector<deepworks::Tensor>& outputs,
                                const std::vector<deepworks::Tensor>& grad_outputs) {
    DeepWorks_Assert(m_impl->m_backend && "Model wasn't compiled !")
    m_impl->m_backend->backward(inputs, outputs, grad_outputs);
}
