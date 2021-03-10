#pragma once

#include <deepworks/layer_info.hpp>
#include <deepworks/placeholder.hpp>

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>
#include <ade/passes/topological_sort.hpp>

namespace deepworks {
namespace graph {

struct Op {
    static const char *name() { return "Op"; }
    LayerInfo info;
};

struct Data {
    static const char *name() { return "Data"; }
    Placeholder ph;
};

struct Type
{
    static const char *name() { return "Type"; }
    enum { OP, DATA } t;
};

using TypedGraph = ade::TypedGraph<Op, Data, Type, ade::passes::TopologicalSortData>;
using Graph      = ade::Graph;

} // namespace graph
} // namespace deepworks
