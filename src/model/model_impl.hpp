#pragma once

#include <vector>
#include <unordered_map>

#include <deepworks/model.hpp>
#include <deepworks/placeholder.hpp>
#include <deepworks/layer.hpp>

#include "model/graph.hpp"
#include "runtime/backend.hpp"

namespace deepworks {

struct deepworks::Model::Impl {
    Impl(Placeholders ins, Placeholders outs);

    graph::Graph            m_graph;
    graph::TypedGraph       m_tgraph;
    deepworks::Placeholders m_inputs;
    deepworks::Placeholders m_outputs;

    using LayerMap = std::unordered_map<std::string, Layer>;
    LayerMap           m_layers_map;
    std::vector<Layer> m_layers;

    std::shared_ptr<IBackend> m_backend;
};

} // namespace deepworks
