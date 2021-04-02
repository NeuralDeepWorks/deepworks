#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPUMaxPooling1x1) {
    std::array<int, 2> kernel{1, 1};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    dw::Placeholder in(dw::Shape{2, 3, 4, 5});
    dw::Model model(in, dw::MaxPooling("max_pool", kernel, padding, stride)(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::testutils::AssertTensorEqual(actual, input);
}

TEST(LayerTests, CPUMaxPooling3x3) {
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{1, 1};
    std::array<int, 2> stride{2, 2};

    dw::Placeholder in(dw::Shape{2, 3, 4, 5});
    dw::Model model(in, dw::MaxPooling("max_pool", kernel, padding, stride)(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    // deepworks::reference::CPUMaxPoolingForward(input.data(),
    //                                            input.shape(),
    //                                            expected.data(),
    //                                            kernel,
    //                                            padding,
    //                                            stride);
    // dw::testutils::AssertTensorEqual(actual, expected);
}

TEST(LayerTests, CPUMaxPooling5x5) {
    std::array<int, 2> kernel{5, 5};
    std::array<int, 2> padding{2, 2};
    std::array<int, 2> stride{3, 3};

    dw::Placeholder in(dw::Shape{5, 3, 25, 26});
    dw::Model model(in, dw::MaxPooling(kernel, padding, stride, "max_pool")(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    // deepworks::reference::CPUMaxPoolingForward(input.data(),
    //                                            input.shape(),
    //                                            expected.data(),
    //                                            kernel,
    //                                            padding,
    //                                            stride);
    // dw::testutils::AssertTensorEqual(actual, expected);
}
