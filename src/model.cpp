#include <iostream>
#include <stack>
#include <unordered_set>

#include <deepworks/model.hpp>

#include "assert.hpp"
#include "call_priv.hpp"
#include "placeholder_priv.hpp"

static deepworks::Model::Unrolled unroll(const deepworks::Placeholders& ins,
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


deepworks::Model::Model(deepworks::Placeholder in,
                        deepworks::Placeholder out)
    : Model(deepworks::Placeholders{in},
            deepworks::Placeholders{out}) {
}

deepworks::Model::Model(std::vector<Placeholder> ins,
                        std::vector<Placeholder> outs)
    : m_tg(m_g), m_inputs(std::move(ins)), m_outputs(std::move(outs)) {
    // NB: Unroll our expression and build computation graph.
    auto unrolled = unroll(m_inputs, m_outputs);
    buildGraph(std::move(unrolled));

    // NB: Sort nodes in graph.
    ade::passes::PassContext context{m_g};
    ade::passes::TopologicalSort()(context);
    auto sorted = m_tg.metadata().get<ade::passes::TopologicalSortData>().nodes();

    // Print the summary, only for debuging ofc
    for (auto&& nh : sorted) {
        switch (m_tg.metadata(nh).get<Type>().t) {
            case Type::OP: {
                auto info = m_tg.metadata(nh).get<Op>().info;
                std::cout << "Layer: " << info.name() << " type: " << info.type() << std::endl;
                break;
            }
            case Type::DATA: {
                auto shape = m_tg.metadata(nh).get<Data>().shape;
                std::cout << "Data: ";
                for (auto&& s : shape) std::cout << s << " ";
                std::cout << std::endl;
                break;
            }
            default:
                DeepWorks_Assert(false);
        }
    }
}

const deepworks::Placeholders& deepworks::Model::inputs()  const {
    return m_inputs;
}

const deepworks::Placeholders& deepworks::Model::outputs() const {
    return m_outputs;
}

const deepworks::Layers& deepworks::Model::layers() const {
    return m_layers;
}

void deepworks::Model::buildGraph(deepworks::Model::Unrolled&& unrolled) {
    std::unordered_map<deepworks::Placeholder::Priv*, ade::NodeHandle> exsisting_data;
    std::unordered_map<deepworks::Call::Priv*       , ade::NodeHandle> exsisting_ops;
    // NB: Link data nodes to their inputs (operations).
    for (auto&& call : unrolled.all_ops) {
        ade::NodeHandle op_nh = m_tg.createNode();
        auto&& call_p = call.priv();
        exsisting_ops.emplace(&call_p, op_nh);
        m_tg.metadata(op_nh).set(Op{call_p.info});
        m_tg.metadata(op_nh).set(Type{Type::OP});

        for (auto&& data : call_p.args) {
            auto&& data_p = data.priv();
            auto it = exsisting_data.find(&data_p);
            if (it == exsisting_data.end()) {
                auto nh = m_tg.createNode();
                // NB: Shapes are copied here, and we can change
                // them according batch size, so this action doesn't
                // affect original placeholder shape.
                m_tg.metadata(nh).set(Data{data.priv().shape});
                m_tg.metadata(nh).set(Type{Type::DATA});
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
            m_tg.metadata(nh).set(Data{data_p.shape});
            m_tg.metadata(nh).set(Type{Type::DATA});
            data_it = exsisting_data.emplace(&data_p, nh).first;
        }

        // NB: Find op handle
        auto&& producer_p = producer.priv();
        auto op_it = exsisting_ops.find(&producer_p);
        // NB: Operation node must be created on the previous step.
        DeepWorks_Assert(op_it != exsisting_ops.end() && "Operation node wasn't found");

        m_tg.link(op_it->second, data_it->second);
    }
}
