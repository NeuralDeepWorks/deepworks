#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPUConvolutionForward1x1) {
    int c_out = 3;
    std::array<int, 2> kernel{1, 1};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    dw::Placeholder in(dw::Shape{2, 3, 4, 5});
    dw::Model model(in, dw::Convolution(c_out, kernel, padding, stride, "conv")(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    auto W = model.layers()[0].params()[0].data();
    auto b = model.layers()[0].params()[1].data();

    // deepworks::reference::CPUConvolutionForward(input.data(),
    //                                             input.shape(),
    //                                             W.data(),
    //                                             b.data(),
    //                                             expected.data(),
    //                                             c_out,
    //                                             kernel,
    //                                             padding,
    //                                             stride);
    // dw::testutils::AssertTensorEqual(actual, expected);
}

TEST(LayerTests, CPUConvolutionForward3x3) {
    int c_out = 32;
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{1, 1};
    std::array<int, 2> stride{2, 2};

    dw::Placeholder in(dw::Shape{5, 3, 11, 16});
    dw::Model model(in, dw::Convolution(c_out, kernel, padding, stride, "conv")(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    auto W = model.layers()[0].params()[0].data();
    auto b = model.layers()[0].params()[1].data();

    // deepworks::reference::CPUConvolutionForward(input.data(),
    //                                             input.shape(),
    //                                             W.data(),
    //                                             b.data(),
    //                                             expected.data(),
    //                                             c_out,
    //                                             kernel,
    //                                             padding,
    //                                             stride);
    // dw::testutils::AssertTensorEqual(actual, expected);
}

TEST(LayerTests, CPUConvolutionBackward1x1) {
    int c_out = 3;
    std::array<int, 2> kernel{1, 1};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    dw::Placeholder in(dw::Shape{2, 3, 4, 5});
    dw::Model model(in, dw::Convolution(c_out, kernel, padding, stride, "conv")(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    std::cout << "input " << input << std::endl;

    dw::Tensor actual(model.outputs()[0].shape());
    dw::Tensor grad_output(actual.shape());
    dw::initializer::uniform(grad_output);

    // forward model to fill im2col_buf
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    auto W = model.layers()[0].params()[0].data();
    auto b = model.layers()[0].params()[1].data();

    model.backward(input, actual, grad_output);

    std::cout << "actual " << actual << std::endl;

    // dw::testutils::AssertTensorEqual(actual, expected);
}

TEST(LayerTests, CPUConvolutionBackward3x3) {
    int c_out = 32;
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{1, 1};
    std::array<int, 2> stride{2, 2};

    dw::Placeholder in(dw::Shape{5, 3, 11, 16});
    dw::Model model(in, dw::Convolution(c_out, kernel, padding, stride, "conv")(in));
    model.compile();

    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    dw::Tensor actual(model.outputs()[0].shape());
    dw::Tensor grad_output(actual.shape());
    dw::initializer::uniform(grad_output);

    // forward model to fill im2col_buf
    model.forward(input, actual);

    dw::Tensor expected(model.outputs()[0].shape());
    auto W = model.layers()[0].params()[0].data();
    auto b = model.layers()[0].params()[1].data();

    model.backward(input, actual, grad_output);

    // dw::testutils::AssertTensorEqual(actual, expected);
}
