#include <stack>
#include <unordered_set>

#include <ade/util/zip_range.hpp>

#include <deepworks/model.hpp>
#include <deepworks/nn.hpp>

#include "expression/call_impl.hpp"
#include "expression/placeholder_impl.hpp"
#include "model/model_impl.hpp"
#include "model/graphbuilder.hpp"
#include "runtime/cpu/cpubackend.hpp"

#include "util/assert.hpp"

#include <iostream>
#include <fstream>

namespace dw = deepworks;
using namespace dw::graph;

struct is_data {
   bool operator()(ade::NodeHandle nh) {
       return tgraph.metadata(nh).get<Type>().t == Type::DATA;
   }

   TypedGraph& tgraph;
};

struct is_op {
   bool operator()(ade::NodeHandle nh) {
       return tgraph.metadata(nh).get<Type>().t == Type::OP;
   }

   TypedGraph& tgraph;
};

// FIXME: This function will be implemented as a graph pass a bit later.
// Let it be a static function.
static void initNodeId(TypedGraph& tgraph) {
    // NB: Should be graph pass as well.
    // NB: Setup data id.
    auto sorted  = tgraph.metadata().get<ade::passes::TopologicalSortData>().nodes();
    for (auto it : ade::util::indexed(ade::util::filter(sorted, is_data{tgraph}))) {
        auto nh  = ade::util::value(it);
        auto idx = ade::util::index(it);
        tgraph.metadata(nh).get<Data>().id = idx;
    }

    // NB: Setup op in/out ids.
    sorted  = tgraph.metadata().get<ade::passes::TopologicalSortData>().nodes();
    for (auto nh : ade::util::filter(sorted, is_op{tgraph})) {
        auto& op = tgraph.metadata(nh).get<deepworks::graph::Op>();

        op.in_ids.resize(nh->inEdges().size());
        for (auto in_eh : nh->inEdges()) {
            auto in_nh      = in_eh->srcNode();
            auto port       = tgraph.metadata(in_eh).get<Port>().p;
            op.in_ids[port] = tgraph.metadata(in_nh).get<Data>().id;
        }

        op.out_ids.resize(nh->outEdges().size());
        for (auto&& out_eh : nh->outEdges()) {
            auto out_nh      = out_eh->dstNode();
            auto port        = tgraph.metadata(out_eh).get<Port>().p;
            op.out_ids[port] = tgraph.metadata(out_nh).get<Data>().id;
        }
    }
}

// FIXME: This function will be implemented as a graph pass a bit later.
// Let it be a static function.
static void createConfig(TypedGraph& tgraph, dw::Model::Config& cfg) {
    auto sorted = tgraph.metadata().get<ade::passes::TopologicalSortData>().nodes();
    for (auto nh : ade::util::filter(sorted, is_data{tgraph})) {
        const auto& d = tgraph.metadata(nh).get<Data>();
        deepworks::Model::Config::PlaceholderInfo ph_info{d.ph.shape(), d.id};
        cfg.ph_map.emplace(d.id, std::move(ph_info));

        if (d.s == Data::Storage::INPUT) {
            cfg.input_ids.push_back(d.id);
        }

        if (d.s == Data::Storage::OUTPUT) {
            cfg.output_ids.push_back(d.id);
        }
    }

    sorted = tgraph.metadata().get<ade::passes::TopologicalSortData>().nodes();
    for (auto nh : ade::util::filter(sorted, is_op{tgraph})) {
        const auto& op = tgraph.metadata(nh).get<Op>();
        deepworks::Model::Config::OperationInfo op_info{op.info.impl().name,
                                                        op.info.impl().type,
                                                        op.info.impl().attrs,
                                                        op.in_ids,
                                                        op.out_ids};
        cfg.sorted_ops.push_back(std::move(op_info));

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
    for (auto nh : ade::util::filter(sorted, is_op{m_tgraph})) {
        // NB: Collect layer inputs.
        Placeholders inputs;
        inputs.reserve(nh->inNodes().size());
        for (auto&& in_nh : nh->inNodes()) {
            inputs.push_back(m_tgraph.metadata(in_nh).get<Data>().ph);
        }

        // NB: Collect layer outputs.
        Placeholders outputs;
        outputs.reserve(nh->outNodes().size());
        for (auto&& out_nh : nh->outNodes()) {
            outputs.push_back(m_tgraph.metadata(out_nh).get<Data>().ph);
        }

        auto&& info = m_tgraph.metadata(nh).get<Op>().info;
        // FIXME: Check that emplace performed without errors. (I'm so lazy)
        auto it = m_layers_map.emplace(info.name(),
                Layer{info, std::move(inputs), std::move(outputs)}).first;
        m_layers.emplace_back(it->second);

        // NB: Collect all parameters from every layer. (Used by optimizer)
        for (auto&& [name, param] : it->second.params()) {
            auto it = m_params.emplace(info.name() + std::string(".") + name, param).first;
            m_state.emplace(it->first, param.data());
        }

        // NB: Push named buffers to state.
        for (auto&& [name, buffer] : it->second.buffers()) {
            m_state.emplace(info.name() + std::string(".") + name, buffer);
        }
    }

    initNodeId(m_tgraph);
    createConfig(m_tgraph, m_cfg);
}

const deepworks::Model::Config& deepworks::Model::cfg() const {
    return m_impl->m_cfg;
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

deepworks::Model deepworks::Model::Build(const deepworks::Model::Config& cfg) {
    std::unordered_map<int, dw::Placeholder> id2ph;
    // NB: First of all create inputs placeholders based on info.
    for (const auto& id : cfg.input_ids) {
        id2ph.emplace(id, cfg.ph_map.at(id).shape);
    }

    // NB: Then go through topologicaly sorted operations
    // and apply their one by one.
    for (const auto& op : cfg.sorted_ops) {
        std::vector<dw::Placeholder> inps;
        for (auto id : op.in_ids) {
            inps.push_back(id2ph.at(id));
        }

        id2ph.emplace(op.out_ids[0],
                dw::make_layer(op.type, op.name, op.attrs, inps));
    }

    // NB: Finally collect model inputs/outputs and create model.
    auto get_placeholder = [&](int id) { return id2ph.at(id); };

    dw::Placeholders inputs;
    ade::util::transform(cfg.input_ids, std::back_inserter(inputs), get_placeholder);

    Placeholders outputs;
    ade::util::transform(cfg.output_ids, std::back_inserter(outputs), get_placeholder);

    return dw::Model(inputs, outputs);
}
