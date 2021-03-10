#pragma once

#include <vector>
#include <unordered_map>

#include "model/graph.hpp"

#include <deepworks/model.hpp>
#include <deepworks/placeholder.hpp>
#include <deepworks/layer.hpp>

namespace deepworks {

class deepworks::Model::Impl {
public:
    Impl(Placeholders ins, Placeholders outs);

    //void buildGraph(Unrolled&& unrolled);

    graph::Graph            m_g;
    graph::TypedGraph       m_tg;
    deepworks::Placeholders m_inputs;
    deepworks::Placeholders m_outputs;

    using LayerMap = std::unordered_map<std::string, Layer>;
    LayerMap           m_layers_map;
    std::vector<Layer> m_layers;
};

} // namespace deepworks
