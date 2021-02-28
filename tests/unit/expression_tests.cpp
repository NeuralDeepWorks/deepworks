// FIXME: Remoe this file after the deepworks::Model is implemented.
// Now it's done only for initial testing internal classes,
// in future only API will be covered.

#include <gtest/gtest.h>

#include <deepworks/deepworks.hpp>

// FIXME: Only for tests
#include "call.hpp"
#include "layer_info.hpp"
#include "placeholder_priv.hpp"
#include "call_priv.hpp"

namespace dw = deepworks;

namespace {

dw::Placeholder Foo(dw::Placeholder in,
                    dw::Shape out_shape,
                    const std::string& name,
                    const std::string& type) {

    dw::Call call{dw::LayerInfo{name, type}};
    call.pass({in});

    // NB: Set the output shape.
    return call.create(std::move(out_shape));
}

dw::Placeholder DeepChain(dw::Placeholder in, int depth) {
    dw::Placeholder curr = in;
    dw::Call last;
    for (int i = 0; i < depth; ++i) {
        in = Foo(in, dw::Shape{i}, "call_" + std::to_string(i), "type");
    }

    return in;
}

} // anonymous namespace

TEST(Placeholder, GetShape) {
    dw::Shape shape{-1, 100};

    dw::Placeholder ph(shape);

    EXPECT_EQ(ph.shape(), shape);
}

TEST(Placeholder, BasicChain) {
    dw::Shape shape{-1, 100};
    dw::Placeholder ph(shape);

    dw::Placeholder out = Foo(ph, dw::Shape{42}, "foo_0", "foo");

    // NB: out is produced by foo operation,
    // so the call isn't empty, let's check it.
    ASSERT_TRUE(out.priv().call);
    // NB: Let's check LayerInfo.
    EXPECT_EQ("foo_0", out.priv().call.value().priv().info.name);
    EXPECT_EQ("foo"  , out.priv().call.value().priv().info.type);
    // NB: In the end, let's check input shape.
    ASSERT_EQ(1u, out.priv().call.value().priv().args.size());
    EXPECT_EQ(shape, out.priv().call.value().priv().args.front().shape());
}

TEST(Placeholder, Overwrite) {
    dw::Shape shape{-1, 100};
    dw::Placeholder ph(shape);

    // NB: Overwrite current placeholder, but keep the chain history.
    ph = Foo(ph, dw::Shape{42}, "foo_0", "foo");

    // NB: The same logic as in BasicChain test
    ASSERT_TRUE(ph.priv().call);
    // NB: Let's check LayerInfo.
    EXPECT_EQ("foo_0", ph.priv().call.value().priv().info.name);
    EXPECT_EQ("foo"  , ph.priv().call.value().priv().info.type);
    // NB: In the end, let's check input shape.
    ASSERT_EQ(1u, ph.priv().call.value().priv().args.size());
    EXPECT_EQ(shape, ph.priv().call.value().priv().args.front().shape());
}

TEST(Placeholder, DeepChain) {
    int depth = 1000;
    dw::Shape shape{-1, 100};
    dw::Placeholder ph(shape);

    ph = DeepChain(ph, depth);

    // NB: Start point
    dw::Placeholder curr = ph;
    for (int i = depth-1; i >= 0; --i) {
        EXPECT_EQ(dw::Shape{i}, curr.shape());
        auto priv = curr.priv();
        EXPECT_TRUE(priv.call);
        auto call = priv.call.value();
        EXPECT_EQ("call_" + std::to_string(i), call.priv().info.name);
        EXPECT_EQ("type", call.priv().info.type);
        // NB: Check number of inputs.
        EXPECT_EQ(1u, call.priv().args.size());
        // NB: Go to the previous node (only single input)
        curr = call.priv().args.front();
    }
    // NB: We reached our input
    EXPECT_EQ(shape, curr.shape());
}
