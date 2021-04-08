#include <numeric>

#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

struct BatchNorm : public ::testing::Test {

    BatchNorm() : in(dw::Shape{batch_size, in_features}),
                  model(in, dw::BatchNorm1D(epsilon, alpha, "batchnorm1d")(in)) {
        model.compile();

        gamma = model.getLayer("batchnorm1d").params()[0].data();
        beta  = model.getLayer("batchnorm1d").params()[1].data();

        gradGamma     = model.getLayer("batchnorm1d").params()[0].grad();
        gradBeta      = model.getLayer("batchnorm1d").params()[1].grad();
        ref_gradGamma = dw::Tensor{gradGamma.shape()};
        ref_gradBeta  = dw::Tensor{gradBeta.shape()};

        dw::initializer::zeros(gradGamma);
        dw::initializer::zeros(gradBeta);
        gradGamma.copyTo(ref_gradGamma);
        gradBeta.copyTo(ref_gradBeta);

        ref_gradInput = dw::Tensor{in.shape()};
        dw::initializer::zeros(ref_gradInput);
    }

    void forward_reference(const dw::Tensor& input, dw::Tensor& output) {
        std::vector<float> ref_moving_mean(in_features);
        std::vector<float> ref_moving_var(in_features);

        dw::reference::CPUBatchNorm1DForward(
                input.data(), output.data(),
                input_centered.data(), std.data(),
                ref_moving_mean.data(), ref_moving_var.data(),
                true, epsilon, alpha,
                gamma.data(), beta.data(), batch_size, in_features);
    }

    void backward_reference(const dw::Tensor& /* input */,
                            const dw::Tensor& /* output */,
                            const dw::Tensor& grad_output) {
        dw::reference::CPUBatchNorm1DBackward(
                input_centered.data(), std.data(),
                grad_output.data(), ref_gradInput.data(),
                gamma.data(), ref_gradGamma.data(), ref_gradBeta.data(),
                batch_size, in_features);
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

    dw::Tensor output{model.outputs()[0].shape()};
    dw::Tensor expected{output.shape()};

    dw::Tensor input_centered{in.shape()};
    dw::Tensor std{output.shape()};

    dw::Tensor ref_gradInput;
};

TEST_F(BatchNorm, Forward) {
    // Init
    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    dw::testutils::AssertTensorEqual(output, expected);
}

TEST_F(BatchNorm, Backward) {
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

    // Assert ??? do we need it? or not this
    // dw::testutils::AssertTensorEqual(?, ref_gradInput);
}
