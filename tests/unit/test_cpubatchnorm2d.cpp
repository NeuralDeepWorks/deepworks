#include <numeric>

#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

struct CPUBatchNorm2DTest : public ::testing::Test {
    CPUBatchNorm2DTest() : in({N, C, H, W}),
                           model(in, dw::BatchNorm2D(epsilon, alpha, "batchnorm2d0")(in)) {
        model.compile();

        gamma = model.getLayer("batchnorm2d0").params().at("gamma").data();
        beta  = model.getLayer("batchnorm2d0").params().at("beta").data();

        gradGamma     = model.getLayer("batchnorm2d0").params().at("gamma").grad();
        gradBeta      = model.getLayer("batchnorm2d0").params().at("beta").grad();
        ref_gradGamma = dw::Tensor{gradGamma.shape()};
        ref_gradBeta  = dw::Tensor{gradBeta.shape()};

        dw::initializer::zeros(gradGamma);
        dw::initializer::zeros(gradBeta);

        gradGamma.copyTo(ref_gradGamma);
        gradBeta.copyTo(ref_gradBeta);

        ref_input_centered = dw::Tensor(in.shape());
        ref_std            = dw::Tensor({N * H * W});

        output   = dw::Tensor{model.outputs()[0].shape()};
        expected = dw::Tensor{output.shape()};
    }

    void forward_reference(const dw::Tensor& input, dw::Tensor& output, bool is_train = true) {
        auto ref_moving_mean = dw::Tensor::zeros({N * H * W});
        auto ref_moving_var  = dw::Tensor::zeros({N * H * W});

        dw::reference::CPUBatchNorm2DForward(input,
                                             expected,
                                             gamma,
                                             beta,
                                             ref_moving_mean,
                                             ref_moving_var,
                                             ref_input_centered,
                                             ref_std,
                                             is_train,
                                             alpha,
                                             epsilon);
    }

    void backward_reference(const dw::Tensor& /* input */,
                            const dw::Tensor& /* output */,
                            const dw::Tensor& grad_output) {
        dw::Tensor ref_gradInput = dw::Tensor{in.shape()};

        dw::reference::CPUBatchNorm2DBackward(
                ref_input_centered, ref_std,
                grad_output,  ref_gradInput,
                gamma, ref_gradGamma, ref_gradBeta);
    }

    int N = 2;
    int C = 3;
    int H = 4;
    int W = 5;

    float epsilon = 0.001;
    float alpha   = 0.05;

    dw::Placeholder in;
    dw::Model       model;

    dw::Tensor gamma, beta;

    dw::Tensor gradGamma, gradBeta;
    dw::Tensor ref_gradGamma, ref_gradBeta;

    dw::Tensor ref_input_centered;
    dw::Tensor ref_std;

    dw::Tensor output;
    dw::Tensor expected;
};

TEST_F(CPUBatchNorm2DTest, ForwardTrain) {
    // Init
    bool is_train = true;
    auto input = dw::Tensor::uniform(in.shape());
    model.train(is_train);

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected, is_train);

    // Assert
    dw::testutils::AssertTensorEqual(output, expected);
}

TEST_F(CPUBatchNorm2DTest, ForwardTest) {
    // Init
    bool is_train = false;
    auto input = dw::Tensor::uniform(in.shape());
    model.train(is_train);

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected, is_train);

    // Assert
    dw::testutils::AssertTensorEqual(output, expected);
}

TEST_F(CPUBatchNorm2DTest, Backward) {
    // Init
    auto input       = dw::Tensor::uniform(in.shape());
    auto grad_output = dw::Tensor::uniform(output.shape());

    // Deepworks
    model.forward(input, output);
    model.backward(input, output, grad_output);

    // Reference
    forward_reference(input, expected);
    backward_reference(input, expected, grad_output);

    // Assert
    dw::testutils::AssertTensorEqual(gradGamma, ref_gradGamma, 1e-4);
    dw::testutils::AssertTensorEqual(gradBeta, ref_gradBeta, 1e-4);
}
