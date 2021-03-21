#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPULinear) {
    int in_features  = 64;
    int out_features = 10;
    int batch_size   = 32;
    dw::Placeholder in(dw::Shape{batch_size, in_features});
    dw::Model model(in, dw::Linear(out_features, "linear")(in));

    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    auto W = model.layers()[0].params()[0].data();
    auto b = model.layers()[0].params()[1].data();

    deepworks::reference::CPULinearForward(input.data(), W.data(), expected.data(),
                                           batch_size, in_features, out_features);

    deepworks::reference::CPULinearAddBias(b.data(), expected.data(), batch_size, out_features);

    dw::testutils::AssertTensorEqual(actual, expected);
}
