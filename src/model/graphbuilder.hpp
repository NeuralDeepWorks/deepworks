#pragma once

#include <unordered_map>

#include <deepworks/call.hpp>

#include "model/graph.hpp"

namespace deepworks {

class GraphBuilder {
public:
    GraphBuilder(graph::Graph& g);
    void build(const Placeholders& ins, const Placeholders& outs);

private:
    struct Unrolled {
        Placeholders all_data;
        Calls        all_ops;
    };
    Unrolled unroll(const Placeholders& ins, const Placeholders& outs);

    ade::NodeHandle getOpNode  (const Call::Impl&  cimpl);
    ade::NodeHandle getDataNode(const Placeholder& ph);

    graph::Graph&     m_g;
    graph::TypedGraph m_tg;

    template <typename T>
    using NodeHandleMap = std::unordered_map<T, ade::NodeHandle>;

    NodeHandleMap<const deepworks::Placeholder::Impl*> m_data;
    NodeHandleMap<const deepworks::Call::Impl*>        m_ops;
};

} // namespace deepworks
