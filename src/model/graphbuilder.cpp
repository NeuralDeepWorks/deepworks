#include <stack>
#include <iostream>

#include "expression/placeholder_impl.hpp"
#include "expression/call_impl.hpp"

#include "model/graphbuilder.hpp"

namespace graph = deepworks::graph;

deepworks::GraphBuilder::GraphBuilder(graph::Graph& g)
    : m_graph(g), m_tgraph(m_graph) {
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
            m_tgraph.link(data_nh, op_nh);
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
        m_tgraph.link(op_nh, data_nh);
    }

    // Get handles associated with plahceholders
    auto extract = [this](const Placeholder& ph) { return m_data[&ph.impl()]; };

    std::vector<ade::NodeHandle> in_handles;
    in_handles.reserve(ins.size());
    std::transform(ins.begin(), ins.end(), std::back_inserter(in_handles), extract);

    std::vector<ade::NodeHandle> out_handles;
    out_handles.reserve(outs.size());
    std::transform(outs.begin(), outs.end(), std::back_inserter(out_handles), extract);

    // Mark input/output nodes.
    auto mark_node = [this](ade::NodeHandle nh, graph::Data::Storage storage) {
        m_tgraph.metadata(nh).get<graph::Data>().s = storage;
    };
    using namespace std::placeholders;
    std::for_each(in_handles.begin(), in_handles.end(),
            std::bind(mark_node, _1, graph::Data::Storage::INPUT));
    std::for_each(out_handles.begin(), out_handles.end(),
            std::bind(mark_node, _1, graph::Data::Storage::OUTPUT));

    // Put graph info
    m_tgraph.metadata().set(graph::Info{in_handles, out_handles, m_data.size()});
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
        ade::NodeHandle nh = m_tgraph.createNode();
        m_ops.emplace(&cimpl, nh);

        graph::Op op;
        op.info = cimpl.info;

        m_tgraph.metadata(nh).set(op);
        m_tgraph.metadata(nh).set(graph::Type{graph::Type::OP});
        it = m_ops.emplace(&cimpl, nh).first;
    }
    return it->second;
}

ade::NodeHandle deepworks::GraphBuilder::getDataNode(const deepworks::Placeholder& ph) {
    auto&& phimpl = ph.impl();
    auto it = m_data.find(&phimpl);
    if (it == m_data.end()) {
        auto nh = m_tgraph.createNode();
    
        graph::Data data;
        data.ph = ph;
        data.s  = graph::Data::Storage::INTERNAL;

        m_tgraph.metadata(nh).set(data);
        m_tgraph.metadata(nh).set(graph::Type{graph::Type::DATA});
        it = m_data.emplace(&phimpl, nh).first;
    }
    return it->second;
}
