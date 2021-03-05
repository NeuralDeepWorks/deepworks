#include <stack>
#include <unordered_set>

#include <deepworks/model.hpp>

#include "expression/call_priv.hpp"
#include "expression/placeholder_priv.hpp"
#include "model/model_priv.hpp"
#include "util/assert.hpp"

#include "runtime/cpu/cpubackend.hpp"

deepworks::Model::Model(deepworks::Placeholder in, deepworks::Placeholder out)
    : deepworks::Model(deepworks::Placeholders{in}, deepworks::Placeholders{out}) {
}

deepworks::Model::Model(deepworks::Placeholders ins, deepworks::Placeholders outs)
    : m_priv(new Priv(std::move(ins), std::move(outs))) {
}

const deepworks::Placeholders& deepworks::Model::inputs() const {
    return m_priv->m_inputs;
}

const deepworks::Placeholders& deepworks::Model::outputs() const {
    return m_priv->m_outputs;
}

const deepworks::Layers& deepworks::Model::layers() const {
    return m_priv->m_layers;
}

deepworks::Layers& deepworks::Model::layers() {
    return m_priv->m_layers;
}

deepworks::Layer deepworks::Model::getLayer(const std::string& name) {
    auto it = m_priv->m_layers_map.find(name);
    DeepWorks_Assert(it != m_priv->m_layers_map.end() && "Layer with that name not found");
    return it->second;
}

void deepworks::Model::compile(int batch_size) {
    return m_priv->compile(batch_size);
}

static deepworks::Unrolled unroll(const deepworks::Placeholders& ins,
                                  const deepworks::Placeholders& outs) {
    deepworks::Placeholders all_data;
    deepworks::Calls        all_ops;

    // NB: Placeholder::Priv is an unique object for every placeholder.
    std::unordered_set<deepworks::Placeholder::Priv*> reached_data;

    std::stack<deepworks::Placeholder> stack;
    for (auto&& out : outs) stack.push(out);

    // NB: Simple DFS implementation.
    // Travers over the graph and collect all data & operation nodes.
    while (!stack.empty()) {
        auto data = stack.top();
        all_data.push_back(data);

        stack.pop();
        // NB: Mark input node as visited.
        reached_data.insert(&data.priv());

        auto has_call = data.priv().call;
        // NB: We reached the node without producer, so we found the input node.
        if (!has_call) {
            continue;
        }

        auto call = has_call.value();
        all_ops.push_back(call);

        auto call_p = call.priv();
        // NB: Go through input placeholders and dive deeper.
        for (auto&& in_ph : call_p.args) {
            if (auto it = reached_data.find(&in_ph.priv()); it == reached_data.end()) {
                stack.push(in_ph);
            }
        }
    }

    return {std::move(all_data), std::move(all_ops)};
}


deepworks::Model::Priv::Priv(deepworks::Placeholders ins,
                             deepworks::Placeholders outs)
    : m_tg(m_g), m_inputs(std::move(ins)), m_outputs(std::move(outs)) {
    // NB: Unroll our expression and build computation graph.
    auto unrolled = unroll(m_inputs, m_outputs);
    buildGraph(std::move(unrolled), m_inputs, m_outputs);

    // NB: Sort nodes in graph.
    ade::passes::PassContext context{m_g};
    ade::passes::TopologicalSort()(context);
    auto sorted = m_tg.metadata().get<ade::passes::TopologicalSortData>().nodes();

    // NB: Create layers for user.
    // It also can a graph pass, but let's keep it separately so far.
    for (auto&& nh : sorted) {
        if (m_tg.metadata(nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::OP) {
            // NB: Collect layer inputs.
            Placeholders inputs;
            inputs.reserve(nh->inNodes().size());
            for (auto&& in_nh : nh->inNodes()) {
                DeepWorks_Assert(m_tg.metadata(in_nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::DATA);
                inputs.push_back(m_tg.metadata(in_nh).get<deepworks::graph::Data>().ph);
            }

            // NB: Collect layer outputs.
            Placeholders outputs;
            outputs.reserve(nh->outNodes().size());
            for (auto&& out_nh : nh->outNodes()) {
                DeepWorks_Assert(m_tg.metadata(out_nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::DATA);
                outputs.push_back(m_tg.metadata(out_nh).get<deepworks::graph::Data>().ph);
            }

            auto&& info = m_tg.metadata(nh).get<deepworks::graph::Op>().info;
            // FIXME: Check that emplace performed without errors. (I'm so lazy)
            auto it = m_layers_map.emplace(info.name(), Layer{info, std::move(inputs), std::move(outputs)}).first;
            m_layers.emplace_back(it->second);
        }
    }

    // NB: Should be graph pass as well.
    initDataID();
}

void deepworks::Model::Priv::buildGraph(deepworks::Unrolled&& unrolled,
                                        deepworks::Placeholders& ins,
                                        deepworks::Placeholders& outs) {
    std::unordered_map<deepworks::Placeholder::Priv*, ade::NodeHandle> exsisting_data;
    std::unordered_map<deepworks::Call::Priv*       , ade::NodeHandle> exsisting_ops;
    // NB: Link data nodes to their inputs (operations).
    for (auto&& call : unrolled.all_ops) {
        ade::NodeHandle op_nh = m_tg.createNode();
        auto&& call_p = call.priv();
        exsisting_ops.emplace(&call_p, op_nh);

        deepworks::graph::Op op;
        op.info = call_p.info;
        m_tg.metadata(op_nh).set(std::move(op));
        m_tg.metadata(op_nh).set(deepworks::graph::Type{deepworks::graph::Type::OP});

        for (auto&& data : call_p.args) {
            auto&& data_p = data.priv();
            auto it = exsisting_data.find(&data_p);
            if (it == exsisting_data.end()) {
                auto nh = m_tg.createNode();
                // NB: Shapes are copied here, and we can change
                // them according batch size, so this action doesn't
                // affect original placeholder shape.
                deepworks::graph::Data d;
                d.ph = data;
                m_tg.metadata(nh).set(std::move(d));
                m_tg.metadata(nh).set(deepworks::graph::Type{deepworks::graph::Type::DATA});
                it = exsisting_data.emplace(&data_p, nh).first;
            }
            // NB: Link operation to input.
            m_tg.link(it->second, op_nh);
        }
    }

    // NB: Now, link data to their producer operation.
    // In current implementation link output placeholders would be enough ?
    for (auto&& data : unrolled.all_data) {
        // NB: Input node was connected on previos step;
        auto&& data_p = data.priv();
        if (!data_p.call) {
            continue;
        }
        auto producer = data_p.call.value();

        // NB: Find data handle
        auto data_it = exsisting_data.find(&data_p);
        if (data_it == exsisting_data.end()) {
            auto nh = m_tg.createNode();
            deepworks::graph::Data d;
            d.ph = data;
            m_tg.metadata(nh).set(std::move(d));
            m_tg.metadata(nh).set(deepworks::graph::Type{deepworks::graph::Type::DATA});
            data_it = exsisting_data.emplace(&data_p, nh).first;
        }

        // NB: Find op handle
        auto&& producer_p = producer.priv();
        auto op_it = exsisting_ops.find(&producer_p);
        // NB: I belive, operation node must be created on the previous step.
        DeepWorks_Assert(op_it != exsisting_ops.end() && "Operation node wasn't found");

        m_tg.link(op_it->second, data_it->second);
    }

    std::vector<ade::NodeHandle> in_nhs;
    in_nhs.reserve(ins.size());
    for (auto&& arg : ins) {
        auto it = exsisting_data.find(&arg.priv());
        DeepWorks_Assert(it != exsisting_data.end() && "Input node wasn't reached while unroll");
        m_tg.metadata(it->second).get<deepworks::graph::Data>().s = deepworks::graph::Data::INPUT;
        in_nhs.push_back(it->second);
    }

    std::vector<ade::NodeHandle> out_nhs;
    out_nhs.reserve(out_nhs.size());
    for (auto&& arg : outs) {
        auto it = exsisting_data.find(&arg.priv());
        DeepWorks_Assert(it != exsisting_data.end() && "Output node wasn't reached while unroll");
        m_tg.metadata(it->second).get<deepworks::graph::Data>().s = deepworks::graph::Data::OUTPUT;
        out_nhs.push_back(it->second);
    }

    m_tg.metadata().set(deepworks::graph::GraphInfo{std::move(in_nhs),
                                                    std::move(out_nhs),
                                                    unrolled.all_data.size()});
}

void deepworks::Model::Priv::initDataID() {
    auto sorted = m_tg.metadata().get<ade::passes::TopologicalSortData>().nodes();
    int id = 0;
    // NB: Setup data id.
    for (auto&& nh : sorted) {
        if (m_tg.metadata(nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::DATA) {
            m_tg.metadata(nh).get<deepworks::graph::Data>().id = id;
            ++id;
        }
    }

    // NB: Setup op in/out ids.
    for (auto&& nh : sorted) {
        if (m_tg.metadata(nh).get<deepworks::graph::Type>().t != deepworks::graph::Type::OP) {
            continue;
        }
        DeepWorks_Assert(m_tg.metadata(nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::OP);
        auto& op = m_tg.metadata(nh).get<deepworks::graph::Op>();

        op.in_ids.reserve(nh->inNodes().size());
        for (auto&& in_nh : nh->inNodes()) {
            DeepWorks_Assert(m_tg.metadata(in_nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::DATA);
            op.in_ids.push_back(m_tg.metadata(in_nh).get<deepworks::graph::Data>().id);
        }

        op.out_ids.reserve(nh->outNodes().size());
        for (auto&& out_nh : nh->outNodes()) {
            DeepWorks_Assert(m_tg.metadata(out_nh).get<deepworks::graph::Type>().t == deepworks::graph::Type::DATA);
            op.out_ids.push_back(m_tg.metadata(out_nh).get<deepworks::graph::Data>().id);
        }
    }
}

void deepworks::Model::Priv::compile(int batch_size) {
    m_backend = std::make_shared<CPUBackend>(m_g, m_tg, batch_size);
}

std::vector<deepworks::Tensor>
deepworks::Model::forward(const std::vector<deepworks::Tensor> & inputs) {
    return m_priv->m_backend->forward(inputs);
}

std::vector<deepworks::Tensor>
deepworks::Model::backward(const std::vector<deepworks::Tensor>& inputs) {
    return m_priv->m_backend->backward(inputs);
}
