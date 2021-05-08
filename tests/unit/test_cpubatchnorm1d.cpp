#include <numeric>

#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

struct CPUBatchNorm1DTest : public ::testing::Test {
    CPUBatchNorm1DTest() : in(dw::Shape{batch_size, in_features}),
                           model(in, dw::BatchNorm1D(epsilon, alpha, "batchnorm1d0")(in)) {
        model.compile();

        gamma = model.getLayer("batchnorm1d0").params().at("gamma").data();
        beta  = model.getLayer("batchnorm1d0").params().at("beta").data();

        gradGamma     = model.getLayer("batchnorm1d0").params().at("gamma").grad();
        gradBeta      = model.getLayer("batchnorm1d0").params().at("beta").grad();
        ref_gradGamma = dw::Tensor{gradGamma.shape()};
        ref_gradBeta  = dw::Tensor{gradBeta.shape()};

        dw::initializer::zeros(gradGamma);
        dw::initializer::zeros(gradBeta);
        gradGamma.copyTo(ref_gradGamma);
        gradBeta.copyTo(ref_gradBeta);

        ref_input_centered = dw::Tensor{in.shape()};
        ref_std = dw::Tensor{dw::Shape{in_features}};

        output = dw::Tensor{model.outputs()[0].shape()};
        expected = dw::Tensor{output.shape()};
    }

    void forward_reference(const dw::Tensor& input, dw::Tensor& output, bool is_train = true) {
        auto ref_running_mean = dw::Tensor::zeros({in_features});
        auto ref_running_var  = dw::Tensor::zeros({in_features});

        dw::reference::CPUBatchNorm1DForward(
                input, output,
                ref_input_centered, ref_std,
                ref_running_mean, ref_running_var,
                is_train, epsilon, alpha,
                gamma, beta);
    }

    void backward_reference(const dw::Tensor& /* input */,
                            const dw::Tensor& /* output */,
                            const dw::Tensor& grad_output) {
        dw::Tensor ref_gradInput = dw::Tensor{in.shape()};

        dw::reference::CPUBatchNorm1DBackward(
                ref_input_centered, ref_std,
                grad_output,  ref_gradInput,
                gamma, ref_gradGamma, ref_gradBeta);
    }

    int batch_size  = 4;
    int in_features = 5;

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

TEST_F(CPUBatchNorm1DTest, ForwardTrain) {
    // Init
    bool is_train = true;
    auto input    = dw::Tensor::uniform(in.shape());

    // Deepworks
    model.train(is_train);
    model.forward(input, output);

    // Reference
    forward_reference(input, expected, is_train);

    // Assert
    dw::testutils::AssertTensorEqual(output, expected);
}

TEST_F(CPUBatchNorm1DTest, ForwardTest) {
    // Init
    bool is_train = false;
    auto input    = dw::Tensor::uniform(in.shape());

    // Deepworks
    model.train(is_train);
    model.forward(input, output);

    // Reference
    forward_reference(input, expected, is_train);

    // Assert
    dw::testutils::AssertTensorEqual(output, expected);
}

TEST_F(CPUBatchNorm1DTest, Backward) {
    // Init
    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);
    dw::Tensor grad_output(output.shape());
    dw::initializer::uniform(grad_output);

    // Deepworks
    model.forward(input, output);
    model.backward(input, output, grad_output);

    // Reference
    forward_reference(input, expected);
    backward_reference(input, expected, grad_output);

    // Assert
    dw::testutils::AssertTensorEqual(gradGamma, ref_gradGamma);
    dw::testutils::AssertTensorEqual(gradBeta, ref_gradBeta);
}
