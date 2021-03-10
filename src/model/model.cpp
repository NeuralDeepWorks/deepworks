#include <stack>
#include <unordered_set>

#include <deepworks/model.hpp>

#include "expression/call_impl.hpp"
#include "expression/placeholder_impl.hpp"

#include "model/model_impl.hpp"
#include "model/graphbuilder.hpp"

#include "util/assert.hpp"

using namespace deepworks::graph;

deepworks::Model::Model(deepworks::Placeholder in, deepworks::Placeholder out)
    : deepworks::Model(deepworks::Placeholders{in}, deepworks::Placeholders{out}) {
}

deepworks::Model::Model(deepworks::Placeholders ins, deepworks::Placeholders outs)
    : m_impl(new Impl(std::move(ins), std::move(outs))) {
}

const deepworks::Placeholders& deepworks::Model::inputs() const {
    return m_impl->m_inputs;
}

const deepworks::Placeholders& deepworks::Model::outputs() const {
    return m_impl->m_outputs;
}

const deepworks::Layers& deepworks::Model::layers() const {
    return m_impl->m_layers;
}

deepworks::Layers& deepworks::Model::layers() {
    return m_impl->m_layers;
}

deepworks::Layer deepworks::Model::getLayer(const std::string& name) {
    auto it = m_impl->m_layers_map.find(name);
    DeepWorks_Assert(it != m_impl->m_layers_map.end() && "Layer with that name not found");
    return it->second;
}

deepworks::Model::Impl::Impl(deepworks::Placeholders ins,
                             deepworks::Placeholders outs)
    : m_tg(m_g), m_inputs(std::move(ins)), m_outputs(std::move(outs)) {
    deepworks::GraphBuilder builder(m_g);
    builder.build(m_inputs, m_outputs);

    // NB: Sort nodes in graph.
    ade::passes::PassContext context{m_g};
    ade::passes::TopologicalSort()(context);
    auto sorted = m_tg.metadata().get<ade::passes::TopologicalSortData>().nodes();

    // NB: Create layers for user.
    // It also can a graph pass, but let's keep it separately so far.
    for (auto&& nh : sorted) {
        if (m_tg.metadata(nh).get<Type>().t == Type::OP) {
            // NB: Collect layer inputs.
            Placeholders inputs;
            inputs.reserve(nh->inNodes().size());
            for (auto&& in_nh : nh->inNodes()) {
                DeepWorks_Assert(m_tg.metadata(in_nh).get<Type>().t == Type::DATA);
                inputs.push_back(m_tg.metadata(in_nh).get<Data>().ph);
            }

            // NB: Collect layer outputs.
            Placeholders outputs;
            outputs.reserve(nh->outNodes().size());
            for (auto&& out_nh : nh->outNodes()) {
                DeepWorks_Assert(m_tg.metadata(out_nh).get<Type>().t == Type::DATA);
                outputs.push_back(m_tg.metadata(out_nh).get<Data>().ph);
            }

            auto&& info = m_tg.metadata(nh).get<Op>().info;
            // FIXME: Check that emplace performed without errors. (I'm so lazy)
            auto it = m_layers_map.emplace(info.name(), Layer{info, std::move(inputs), std::move(outputs)}).first;
            m_layers.emplace_back(it->second);
        }
    }
}
