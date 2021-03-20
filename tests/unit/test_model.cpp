#include <gtest/gtest.h>

#include <deepworks/deepworks.hpp>

namespace dw = deepworks;

// FIXME: There is should be more tests !!!
TEST(Model, SimpleModel) {
    // NB: Define our MNIST winner.
    dw::Placeholder in(dw::Shape{-1, 100});

    auto out = dw::Linear(50, "linear_0")(in);
    out = dw::ReLU("relu_1")(out);
    out = dw::Linear(10, "linear_2")(out);
    out = dw::Softmax("probs")(out);
    
    dw::Model model(in, out);

    // NB: (bias + weight) * 2
    EXPECT_EQ(4u, model.params().size());

    // NB: Check model inputs/outputs.
    EXPECT_EQ(1u, model.inputs().size());
    EXPECT_EQ(1u, model.outputs().size());
    EXPECT_EQ(in.shape(),  model.inputs()[0].shape());
    EXPECT_EQ(out.shape(), model.outputs()[0].shape());

    // NB: Check model per-layer.
    // Layers are topological-sorted.
    const auto& layers = model.layers();
    ASSERT_EQ(4u, layers.size());
    EXPECT_EQ(1u, layers[0].inputs().size());
    EXPECT_EQ(1u, layers[0].outputs().size());
    EXPECT_EQ("linear_0", layers[0].name());
    EXPECT_EQ("Linear"  , layers[0].type());

    EXPECT_EQ(1u, layers[1].inputs().size());
    EXPECT_EQ(1u, layers[1].outputs().size());
    EXPECT_EQ("relu_1", layers[1].name());
    EXPECT_EQ("ReLU"  , layers[1].type());

    EXPECT_EQ(1u, layers[2].inputs().size());
    EXPECT_EQ(1u, layers[2].outputs().size());
    EXPECT_EQ("linear_2", layers[2].name());
    EXPECT_EQ("Linear"  , layers[2].type());

    EXPECT_EQ(1u, layers[3].inputs().size());
    EXPECT_EQ(1u, layers[3].outputs().size());
    EXPECT_EQ("probs"  , layers[3].name());
    EXPECT_EQ("Softmax", layers[3].type());

    // NB: Now let's check what happpend
    // if we wanna get layer by name.
    auto linear_0 = model.getLayer("linear_0");
    auto relu_1   = model.getLayer("relu_1");
    auto linear_2 = model.getLayer("linear_2");
    auto probs    = model.getLayer("probs");

    EXPECT_EQ(1u, linear_0.inputs().size());
    EXPECT_EQ(1u, linear_0.outputs().size());
    EXPECT_EQ("linear_0", linear_0.name());
    EXPECT_EQ("Linear"  , linear_0.type());

    EXPECT_EQ(1u, relu_1.inputs().size());
    EXPECT_EQ(1u, relu_1.outputs().size());
    EXPECT_EQ("relu_1", relu_1.name());
    EXPECT_EQ("ReLU"  , relu_1.type());

    EXPECT_EQ(1u, linear_2.inputs().size());
    EXPECT_EQ(1u, linear_2.outputs().size());
    EXPECT_EQ("linear_2", linear_2.name());
    EXPECT_EQ("Linear"  , linear_2.type());

    EXPECT_EQ(1u, probs.inputs().size());
    EXPECT_EQ(1u, probs.outputs().size());
    EXPECT_EQ("probs"  , probs.name());
    EXPECT_EQ("Softmax", probs.type());

    // NB: Finnaly let's get the nonexistent layer.
    EXPECT_ANY_THROW(model.getLayer("linear_3"));
}
