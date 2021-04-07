#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPUMaxPoolingForward1x1) {
    std::array<int, 2> kernel{1, 1};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    dw::Placeholder in(dw::Shape{2, 3, 4, 5});
    dw::Model model(in, dw::MaxPooling(kernel, padding, stride, "max_pool")(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::testutils::AssertTensorEqual(actual, input);
}

TEST(LayerTests, CPUMaxPoolingForward3x3) {
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{1, 1};
    std::array<int, 2> stride{2, 2};

    dw::Placeholder in(dw::Shape{2, 3, 4, 5});
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

TEST(LayerTests, CPUMaxPoolingForward5x5) {
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

TEST(LayerTests, CPUMaxPoolingBackward1x1) {
    std::array<int, 2> kernel{1, 1};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    dw::Placeholder in(dw::Shape{2, 3, 4, 5});
    dw::Model model(in, dw::MaxPooling(kernel, padding, stride, "max_pool")(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    dw::Tensor grad_output(actual.shape());
    dw::initializer::uniform(grad_output);

    // forward model to fill max_indices
    model.forward(input, actual);
    model.backward(input, actual, grad_output);
}

TEST(LayerTests, CPUMaxPoolingBackward3x3Simple) {
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    dw::Placeholder in(dw::Shape{1, 1, 3, 3});
    dw::Model model(in, dw::MaxPooling(kernel, padding, stride, "max_pool")(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    dw::Tensor grad_output(actual.shape());
    dw::initializer::uniform(grad_output);

    // forward model to fill max_indices
    model.forward(input, actual);
    model.backward(input, actual, grad_output);
}

TEST(LayerTests, CPUMaxPoolingBackward3x3) {
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{1, 1};
    std::array<int, 2> stride{2, 2};

    dw::Placeholder in(dw::Shape{2, 3, 4, 5});
    dw::Model model(in, dw::MaxPooling(kernel, padding, stride, "max_pool")(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    dw::Tensor grad_output(actual.shape());
    dw::initializer::uniform(grad_output);

    // forward model to fill max_indices
    model.forward(input, actual);

    model.backward(input, actual, grad_output);
}

TEST(LayerTests, CPUMaxPoolingBackward5x5) {
    std::array<int, 2> kernel{5, 5};
    std::array<int, 2> padding{2, 2};
    std::array<int, 2> stride{3, 3};

    dw::Placeholder in(dw::Shape{5, 3, 25, 26});
    dw::Model model(in, dw::MaxPooling(kernel, padding, stride, "max_pool")(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    dw::Tensor grad_output(actual.shape());
    dw::initializer::uniform(grad_output);

    // forward model to fill max_indices
    model.forward(input, actual);

    model.backward(input, actual, grad_output);
}
