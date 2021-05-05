#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPUGlobalAvgPoolForward) {
    dw::Placeholder in({2, 3, 9, 10});
    dw::Model model(in, dw::GlobalAvgPooling("global_avg_pool")(in));

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input, -2.0f, 2.0f);

    model.compile();

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    deepworks::reference::CPUGlobalAvgPoolingForward(input, expected);

    dw::testutils::AssertTensorEqual(actual, expected);
}
