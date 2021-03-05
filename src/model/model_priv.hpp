#pragma once

#include <vector>
#include <unordered_map>

#include <deepworks/model.hpp>
#include <deepworks/placeholder.hpp>
#include <deepworks/layer.hpp>

#include "expression/call.hpp"
#include "model/graph.hpp"
#include "runtime/backend.hpp"

namespace deepworks {

struct Unrolled {
    Placeholders all_data;
    Calls        all_ops;
};

class deepworks::Model::Priv {
public:
    Priv(Placeholders ins, Placeholders outs);

    void compile(int batch_size);
    // FIXME: ins/outs should be const.
    void buildGraph(Unrolled&& unrolled,
                    Placeholders& ins,
                    Placeholders& outs);
    void initDataID();

    // NB: So the Model is quite big.
    graph::Graph      m_g;
    graph::TypedGraph m_tg;
    deepworks::Placeholders m_inputs;
    deepworks::Placeholders m_outputs;

    using LayerMap = std::unordered_map<std::string, Layer>;
    LayerMap           m_layers_map;
    std::vector<Layer> m_layers;

    std::shared_ptr<IBackend> m_backend;
};

} // namespace deepworks
