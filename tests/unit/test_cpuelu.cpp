#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPUELUForward) {
    int in_features = 100;
    int batch_size  = 32;
    float alpha = 0.2f;
    dw::Placeholder in(dw::Shape{batch_size, in_features});
    dw::Model model(in, dw::ELU(alpha, "elu")(in));

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input, -2.0f, 2.0f);

    model.compile();

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    deepworks::reference::CPUELUForward(input, expected, alpha);

    dw::testutils::AssertTensorEqual(actual, expected);
}
