#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPUDropoutForward) {
    int in_features = 100;
    int batch_size  = 32;
    float p = 0.2f;
    dw::Placeholder in(dw::Shape{batch_size, in_features});
    dw::Model model(in, dw::Dropout(p, "Dropout")(in));

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input, -2.0f, 2.0f);

    model.compile();

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());

    auto mask = model.layers()[0].params().at("mask").data();
    bool is_train =  model.layers()[0].params().at("mask").is_trainable();
    deepworks::reference::CPUDropoutForward(input, mask, expected, p, is_train);

    dw::testutils::AssertTensorEqual(actual, expected);
}
