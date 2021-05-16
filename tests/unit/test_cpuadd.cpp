#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPUAddTest) {
    int in_features = 100;
    int batch_size  = 32;

    dw::Placeholder in0(dw::Shape{batch_size, in_features});
    dw::Placeholder in1(dw::Shape{batch_size, in_features});

    dw::Model model({in0, in1}, {dw::Add()(in0, in1)});

    auto input0 = dw::Tensor::uniform(in0.shape());
    auto input1 = dw::Tensor::uniform(in1.shape());
    std::vector<dw::Tensor> actual{dw::Tensor(model.outputs()[0].shape())};

    model.compile();

    model.forward({input0, input1}, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    deepworks::reference::CPUAddForward(input0, input1, expected);

    dw::testutils::AssertTensorEqual(actual.at(0), expected);
}
