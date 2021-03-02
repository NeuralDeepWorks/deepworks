#include <iostream>
#include <stack>

#include <deepworks/model.hpp>

#include "assert.hpp"
#include "call_priv.hpp"
#include "placeholder_priv.hpp"

deepworks::Model::Model(deepworks::Placeholder in,
                        deepworks::Placeholder out)
    : Model(deepworks::Placeholders{in},
            deepworks::Placeholders{out}) {
}

deepworks::Model::Model(std::vector<Placeholder> ins,
                        std::vector<Placeholder> outs)
    : m_inputs(std::move(ins)), m_outputs(std::move(outs)) {
        unroll(m_inputs, m_outputs);
}

const deepworks::Placeholders& deepworks::Model::inputs()  const {
    return m_inputs;
}

const deepworks::Placeholders& deepworks::Model::outputs() const {
    return m_outputs;
}

const deepworks::Layers& deepworks::Model::layers() const {
    return m_layers;
}

void deepworks::Model::unroll(const deepworks::Placeholders& ins,
                              const deepworks::Placeholders& outs) {
    DeepWorks_Assert(ins.size()  == 1u && "Only single input for model is supported now");
    DeepWorks_Assert(outs.size() == 1u && "Only single output for model is supported now");

    // FIXME: It should have more complex logic.
    std::stack<Layer> stack;
    auto iter = outs[0];
    while (true) {
        auto priv = iter.priv();
        // NB: Found input, stop unroll
        if (!priv.call) {
            break;
        }

        auto call = priv.call.value();
        auto info = call.priv().info;

        stack.push(Layer{info, call.priv().args, {iter}});

        DeepWorks_Assert(call.priv().args.size() == 1u
                && "Only single input operations are supported now");

        iter = call.priv().args.front();
    }

    m_layers.reserve(stack.size());
    while (!stack.empty()) {
        auto& top = stack.top();
        m_layers.push_back(std::move(top));
        stack.pop();
    }
}
