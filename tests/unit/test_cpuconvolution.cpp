#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

struct CPUConvolutionModelTest: public ::testing::Test {
    void testConvolution(int c_out,
                         const std::array<int, 2>& kernel,
                         const std::array<int, 2>& padding,
                         const std::array<int, 2>& stride,
                         const dw::Shape& shape) {
        dw::Placeholder in(shape);
        dw::Model model(in, dw::Convolution(c_out, kernel, padding, stride, "conv")(in));
        model.compile();

        auto input = dw::Tensor::uniform(model.inputs()[0].shape());

        dw::Tensor actual(model.outputs()[0].shape());
        model.forward(input, actual);

        dw::Tensor expected(model.outputs()[0].shape());
        auto W = model.layers()[0].params().at("weight").data();
        auto b = model.layers()[0].params().at("bias").data();

        dw::reference::CPUConvolution2DForward(input,
                                               W,
                                               b,
                                               expected,
                                               kernel,
                                               padding,
                                               stride);

        auto grad_W = model.layers()[0].params().at("weight").grad();
        auto grad_b = model.layers()[0].params().at("bias").grad();

        auto grad_output = dw::Tensor::uniform(expected.shape());
        dw::Tensor grad_input(shape);

        model.backward(input, actual, grad_output);

        dw::Tensor ref_grad_W(grad_W.shape());
        dw::Tensor ref_grad_b(grad_b.shape());

        dw::reference::CPUConvolution2DBackward(input,
                                                grad_output,
                                                W,
                                                b,
                                                ref_grad_W,
                                                ref_grad_b,
                                                grad_input,
                                                kernel,
                                                padding,
                                                stride);
        float threshold = 1e-4;
        dw::testutils::AssertTensorEqual(actual, expected, threshold);
        dw::testutils::AssertTensorEqual(grad_W, ref_grad_W, threshold);
        dw::testutils::AssertTensorEqual(grad_b, ref_grad_b, threshold);
    }
};

TEST_F(CPUConvolutionModelTest, CPUConvolutionTEMP) {
    int c_out = 3;
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{1, 1};
    std::array<int, 2> stride{1, 1};

    testConvolution(c_out, kernel, padding, stride, dw::Shape{2, 3, 4, 4});
}

TEST_F(CPUConvolutionModelTest, CPUConvolution1x1) {
    int c_out = 3;
    std::array<int, 2> kernel{1, 1};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    testConvolution(c_out, kernel, padding, stride, dw::Shape{2, 3, 6, 7});
}

TEST_F(CPUConvolutionModelTest, CPUConvolution4x5) {
    int c_out = 3;
    std::array<int, 2> kernel{4, 5};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    testConvolution(c_out, kernel, padding, stride, dw::Shape{2, 3, 4, 5});
}

TEST_F(CPUConvolutionModelTest, CPUConvolution1x1_simple) {
    int c_out = 3;
    std::array<int, 2> kernel{1, 1};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    testConvolution(c_out, kernel, padding, stride, dw::Shape{1, 1, 1, 1});
}

TEST_F(CPUConvolutionModelTest, CPUConvolution3x3) {
    int c_out = 64;
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{1, 1};
    std::array<int, 2> stride{1, 1};

    testConvolution(c_out, kernel, padding, stride, dw::Shape{1, 5, 7, 11});
}

TEST_F(CPUConvolutionModelTest, CPUConvolution3x3_pad0x0) {
    int c_out = 16;
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    testConvolution(c_out, kernel, padding, stride, dw::Shape{3, 2, 5, 8});
}

TEST_F(CPUConvolutionModelTest, CPUConvolution3x3_stride2x2) {
    int c_out = 14;
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{1, 1};
    std::array<int, 2> stride{2, 2};

    //testConvolution(c_out, kernel, padding, stride, dw::Shape{2, 8, 7, 9});
    testConvolution(c_out, kernel, padding, stride, dw::Shape{1, 1, 7, 9});
}

TEST_F(CPUConvolutionModelTest, CPUConvolution5x5) {
    int c_out = 1;
    std::array<int, 2> kernel{5, 5};
    std::array<int, 2> padding{2, 2};
    std::array<int, 2> stride{2, 2};

    testConvolution(c_out, kernel, padding, stride, dw::Shape{2, 8, 15, 13});
}
