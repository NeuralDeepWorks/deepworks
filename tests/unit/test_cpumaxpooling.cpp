#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

struct CPUMaxPoolingModelTest: public ::testing::Test {
    void testMaxPoolForward(const std::array<int, 2>& kernel,
              const std::array<int, 2>& padding,
              const std::array<int, 2>& stride,
              const dw::Shape& shape) {
        dw::Placeholder in(shape);
        dw::Model model(in, dw::MaxPooling(kernel, padding, stride, "pool")(in));
        model.compile();

        dw::Tensor input(model.inputs()[0].shape());
        dw::initializer::uniform(input);

        dw::Tensor actual(model.outputs()[0].shape());
        model.forward(input, actual);

        dw::Tensor expected(model.outputs()[0].shape());
        dw::reference::CPUMaxPooling2DForward(input,
                                            expected,
                                            kernel,
                                            padding,
                                            stride);
        dw::testutils::AssertTensorEqual(actual, expected);

        dw::Tensor grad_output(actual.shape());
        dw::initializer::uniform(grad_output);

        // check backward will be called without errors
        model.backward(input, actual, grad_output);
    }
};

TEST_F(CPUMaxPoolingModelTest, CPUMaxPoolingForward1x1) {
    std::array<int, 2> kernel{1, 1};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{1, 1};

    testMaxPoolForward(kernel, padding, stride, dw::Shape{2, 3, 4, 5});
}

TEST_F(CPUMaxPoolingModelTest, CPUMaxPoolingForward2x2) {
    std::array<int, 2> kernel{2, 2};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{2, 2};

    testMaxPoolForward(kernel, padding, stride, dw::Shape{4, 16, 17, 35});
}
TEST_F(CPUMaxPoolingModelTest, CPUMaxPoolingForward3x3) {
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{0, 0};
    std::array<int, 2> stride{2, 2};

    testMaxPoolForward(kernel, padding, stride, dw::Shape{2, 2, 3, 15});
}

TEST_F(CPUMaxPoolingModelTest, CPUMaxPoolingForward3x3_strides) {
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{1, 1};
    std::array<int, 2> stride{2, 2};

    testMaxPoolForward(kernel, padding, stride, dw::Shape{3, 1, 13, 12});
}

TEST_F(CPUMaxPoolingModelTest, CPUMaxPoolingForward3x3_pads) {
    std::array<int, 2> kernel{3, 3};
    std::array<int, 2> padding{1, 1};
    std::array<int, 2> stride{1, 1};

    testMaxPoolForward(kernel, padding, stride, dw::Shape{4, 8, 14, 15});
}

TEST_F(CPUMaxPoolingModelTest, CPUMaxPoolingForward5x5) {
    std::array<int, 2> kernel{5, 5};
    std::array<int, 2> padding{2, 2};
    std::array<int, 2> stride{2, 2};

    testMaxPoolForward(kernel, padding, stride, dw::Shape{4, 8, 24, 35});
}
