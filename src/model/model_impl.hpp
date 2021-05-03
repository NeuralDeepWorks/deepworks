#pragma once

#include <vector>
#include <unordered_map>

#include <deepworks/model.hpp>
#include <deepworks/placeholder.hpp>
#include <deepworks/layer.hpp>
#include <deepworks/parameter.hpp>

#include "model/graph.hpp"
#include "runtime/backend.hpp"

#include "model/graph.hpp"
#include "runtime/backend.hpp"

namespace deepworks {

struct Model::Impl {
    Impl(Placeholders ins, Placeholders outs);

    graph::Graph            m_graph;
    graph::TypedGraph       m_tgraph;
    Placeholders            m_inputs;
    Placeholders            m_outputs;
    Model::Config           m_cfg;

    using LayerMap = std::unordered_map<std::string, Layer>;
    LayerMap           m_layers_map;
    std::vector<Layer> m_layers;
    ParamMap           m_params;
    StateDict          m_state;

    std::shared_ptr<IBackend> m_backend;
};

} // namespace deepworks
