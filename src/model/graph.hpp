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

    std::vector<int> in_ids;
    std::vector<int> out_ids;
    LayerInfo        info;
};

struct Data {
    static const char *name() { return "Data"; }

    enum Storage {
        INPUT,
        OUTPUT,
        INTERNAL
    };

    Storage s;
    int     id;

    // FIXME: This should be outside of the Data structure.
    // see below.
    Placeholder ph;
};

struct Type
{
    static const char *name() { return "Type"; }
    enum { OP, DATA } t;
};

// NB: Some useful information about the graph.
struct Info {
    static const char *name() { return "Info"; }

    std::vector<ade::NodeHandle> in_nhs;
    std::vector<ade::NodeHandle> out_nhs;
    size_t                       num_data_nodes;
};

using TypedGraph = ade::TypedGraph<Op, Data, Type, Info, ade::passes::TopologicalSortData>;
using Graph      = ade::Graph;

} // namespace graph
} // namespace deepworks
