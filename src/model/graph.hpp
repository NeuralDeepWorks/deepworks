#pragma once

#include <deepworks/placeholder.hpp>
#include <deepworks/layer_info.hpp>

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

    enum Storage {INPUT, OUTPUT, INTERNAL} s;
    int         id;
    // FIXME: This should be outside of the Data structure.
    // see below.
    Placeholder ph;
};

//struct Some {
    //Placeholder ph;
//}

struct Type {
    static const char *name() { return "Type"; }
    enum { OP, DATA } t;
};

// NB: Some useful information about graph in general.
struct GraphInfo {
    static const char *name() { return "GraphInfo"; }

    std::vector<ade::NodeHandle> in_nhs;
    std::vector<ade::NodeHandle> out_nhs;
    size_t                       num_data_nodes;
};

using TypedGraph = ade::TypedGraph<Op, Data, Type, GraphInfo, ade::passes::TopologicalSortData>;
using Graph      = ade::Graph;

} // namespace graph
} // namespace deepworks
