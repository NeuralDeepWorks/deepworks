#pragma once

#include <vector>
#include <unordered_map>

#include <deepworks/model.hpp>
#include <deepworks/placeholder.hpp>
#include <deepworks/layer.hpp>

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>
#include <ade/passes/topological_sort.hpp>

namespace deepworks {

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

struct Unrolled {
    Placeholders all_data;
    Calls        all_ops;
};

class deepworks::Model::Impl {
public:
    Impl(Placeholders ins, Placeholders outs);

    void buildGraph(Unrolled&& unrolled);

    using TypedGr = ade::TypedGraph<Op, Data, Type, ade::passes::TopologicalSortData>;
    ade::Graph              m_g;
    TypedGr                 m_tg;
    deepworks::Placeholders m_inputs;
    deepworks::Placeholders m_outputs;

    using LayerMap = std::unordered_map<std::string, Layer>;
    LayerMap           m_layers_map;
    std::vector<Layer> m_layers;
};

} // namespace deepworks
