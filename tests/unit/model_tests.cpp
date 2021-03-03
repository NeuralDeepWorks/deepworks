#include <gtest/gtest.h>

#include <deepworks/model.hpp>
#include <deepworks/layers.hpp>

namespace dw = deepworks;

namespace {

void print(const dw::Model& m) {
    std::cout << "=======================" << std::endl;
    for (auto&& l : m.layers()) {
        std::cout << "Layer: " << l.name() << std::endl;
    }
    std::cout << "=======================\n" << std::endl;
}

}

TEST(Model, SimpleModel) {
    // NB: Define our MNIST winner.
    dw::Placeholder in(dw::Shape{-1, 100});

    auto out = dw::Linear(50, "linear_0")(in);
    out = dw::ReLU("relu_1")(out);
    out = dw::Linear(10, "linear_2")(out);
    out = dw::Softmax("probs")(out);
    
    dw::Model base(in, out);

    EXPECT_EQ(1u, base.inputs().size());
    EXPECT_EQ(1u, base.outputs().size());

    EXPECT_EQ(in.shape(),  base.inputs()[0].shape());
    EXPECT_EQ(out.shape(), base.outputs()[0].shape());

    //print(base);

    //auto x = dw::Linear(10, "linear_3")(base.outputs()[0]);

    //dw::Model model(base.inputs()[0], x);

    //print(model);
}
