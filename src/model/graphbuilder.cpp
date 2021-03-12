#include <stack>
#include <iostream>

#include "expression/placeholder_impl.hpp"
#include "expression/call_impl.hpp"

#include "model/graphbuilder.hpp"

deepworks::GraphBuilder::GraphBuilder(graph::Graph& g)
    : m_g(g), m_tg(m_g) {
}

void deepworks::GraphBuilder::build(const deepworks::Placeholders& ins,
                                    const deepworks::Placeholders& outs) {
    auto unrolled = unroll(ins, outs);
    // NB: Link data nodes to their inputs (operations).
    for (auto&& call : unrolled.all_ops) {
        auto&& cimpl = call.impl();
        auto op_nh   = getOpNode(cimpl);

        for (auto&& ph : cimpl.args) {
            auto data_nh = getDataNode(ph);
            // NB: Link data to operation.
            m_tg.link(data_nh, op_nh);
        }
    }

    // NB: Now, link data to their producer operation.
    // In current implementation link output placeholders would be enough ?
    for (auto&& ph : unrolled.all_data) {
        // NB: Input node was connected on previous step.
        auto&& phimpl = ph.impl();
        if (!phimpl.call) {
            continue;
        }

        auto call    = phimpl.call.value();
        auto data_nh = getDataNode(ph);
        auto op_nh   = getOpNode(call.impl());
        m_tg.link(op_nh, data_nh);
    }
}

deepworks::GraphBuilder::Unrolled deepworks::GraphBuilder::unroll(const deepworks::Placeholders& ins,
                                                                  const deepworks::Placeholders& outs) {
    deepworks::Placeholders all_data;
    deepworks::Calls        all_ops;

    // NB: Placeholder::Impl is an unique object for every placeholder.
    std::unordered_set<deepworks::Placeholder::Impl*> reached_data;

    std::stack<deepworks::Placeholder> stack;
    for (auto&& out : outs) stack.push(out);

    // NB: Simple DFS implementation.
    // Travers over the graph and collect all data & operation nodes.
    while (!stack.empty()) {
        auto data = stack.top();
        all_data.push_back(data);

        stack.pop();
        // NB: Mark input node as visited.
        reached_data.insert(&data.impl());

        auto has_call = data.impl().call;
        // NB: We reached the node without producer, so we found the input node.
        if (!has_call) {
            continue;
        }

        auto call = has_call.value();
        all_ops.push_back(call);

        auto call_p = call.impl();
        // NB: Go through input placeholders and dive deeper.
        for (auto&& in_ph : call_p.args) {
            if (auto it = reached_data.find(&in_ph.impl()); it == reached_data.end()) {
                stack.push(in_ph);
            }
        }
    }

    return {std::move(all_data), std::move(all_ops)};
}

ade::NodeHandle deepworks::GraphBuilder::getOpNode(const deepworks::Call::Impl& cimpl) {
    auto it = m_ops.find(&cimpl);
    if (it == m_ops.end()) {
        ade::NodeHandle nh = m_tg.createNode();
        m_ops.emplace(&cimpl, nh);

        deepworks::graph::Op op;
        op.info = cimpl.info;

        m_tg.metadata(nh).set(op);
        m_tg.metadata(nh).set(deepworks::graph::Type{deepworks::graph::Type::OP});
        it = m_ops.emplace(&cimpl, nh).first;
    }
    return it->second;
}

ade::NodeHandle deepworks::GraphBuilder::getDataNode(const deepworks::Placeholder& ph) {
    auto&& phimpl = ph.impl();
    auto it = m_data.find(&phimpl);
    if (it == m_data.end()) {
        auto nh = m_tg.createNode();
    
        deepworks::graph::Data data;
        data.ph = ph;

        m_tg.metadata(nh).set(data);
        m_tg.metadata(nh).set(deepworks::graph::Type{deepworks::graph::Type::DATA});
        it = m_data.emplace(&phimpl, nh).first;
    }
    return it->second;
}
