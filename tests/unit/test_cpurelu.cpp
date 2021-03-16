#include <random>

#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

// FIXME: It's initial tests implementation,
// will be more generic and elegant in future.
TEST(LayerTests, CPUReLU) {
    int n_features = 100;
    dw::Placeholder in(dw::Shape{1, n_features});
    dw::Model model(in, dw::ReLU("relu")(in));

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    model.compile();

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    deepworks::reference::CPUReLUForward(input.data(), expected.data(), input.total());

    dw::testutils::AssertTensorEqual(actual, expected);
}
