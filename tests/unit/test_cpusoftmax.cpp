#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPUSoftmax) {
    int batch_size = 32;
    int num_features = 100;
    dw::Placeholder in(dw::Shape{batch_size, num_features});
    dw::Model model(in, dw::Softmax("softmax")(in));

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    model.compile();

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    deepworks::reference::CPUSoftmaxForward(input.data(), expected.data(), input.shape()[0], input.shape()[1]);

    dw::testutils::AssertTensorEqual(actual, expected);
}
