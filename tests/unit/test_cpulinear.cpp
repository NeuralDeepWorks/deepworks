#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

struct CPULinearModelTest: public ::testing::Test {
    CPULinearModelTest()
        : in(dw::Shape{batch_size, in_features}),
          model(in, dw::Linear(out_features, "linear")(in))  {
        model.compile();
    }

    int in_features  = 64;
    int out_features = 10;
    int batch_size   = 32;

    dw::Placeholder in;
    dw::Model model;
};

TEST_F(CPULinearModelTest, CPULinearForward) {
    dw::Tensor X_train(in.shape());
    dw::initializer::uniform(X_train);
    auto W = model.layers()[0].params().at("weight").data();
    auto b = model.layers()[0].params().at("bias").data();
    dw::initializer::uniform(b);

    dw::Tensor actual(model.outputs()[0].shape());
    model.forward(X_train, actual);

    dw::Tensor expected(model.outputs()[0].shape());

    deepworks::reference::CPULinearForward(X_train.data(), W.data(), expected.data(),
                                           batch_size, in_features, out_features);

    deepworks::reference::CPULinearAddBias(b.data(), expected.data(), batch_size, out_features);

    dw::testutils::AssertTensorEqual(actual, expected);
}

TEST_F(CPULinearModelTest, CPULinearBackward) {
    dw::Tensor X_train(in.shape());
    dw::initializer::uniform(X_train);

    dw::Tensor predict(model.outputs()[0].shape());
    dw::Tensor grad_output(predict.shape());
    dw::initializer::uniform(grad_output);

    model.backward(X_train, predict, grad_output);

    // Reference
    auto W = model.layers()[0].params().at("weight").data();
    auto b = model.layers()[0].params().at("bias").data();
    auto Wgrad = model.layers()[0].params().at("weight").grad();
    auto bgrad = model.layers()[0].params().at("bias").grad();
    dw::Tensor expected_Wgrad(Wgrad.shape());
    dw::Tensor expected_bgrad(bgrad.shape());
    dw::Tensor grad_input(in.shape());

    dw::reference::CPULinearBackward(X_train.data(), W.data(), grad_output.data(),
                                     expected_Wgrad.data(), grad_input.data(),
                                     batch_size, in_features, out_features);

    dw::reference::CPULinearBiasBackward(grad_output.data(), expected_bgrad.data(),
                                         batch_size, out_features);

    dw::testutils::AssertTensorEqual(expected_Wgrad, Wgrad);
    dw::testutils::AssertTensorEqual(expected_bgrad, bgrad);
}

struct CPULinearAfterConvolutionModelTest : public ::testing::Test {
    CPULinearAfterConvolutionModelTest()
            : in(dw::Shape{batch_size, in_channels, in_features, in_features}),
              model(buildModel()) {
        model.compile();

	W_conv  = model.getLayer("conv1").params().at("weight").data();
        b_conv  = model.getLayer("conv1").params().at("bias").data();
        W       = model.getLayer("linear2").params().at("weight").data();
        b       = model.getLayer("linear2").params().at("bias").data();

        expected_W_conv  = dw::Tensor{W_conv.shape()};
        expected_b_conv  = dw::Tensor{b_conv.shape()};
        expected_W       = dw::Tensor{W.shape()};
        expected_b       = dw::Tensor{b.shape()};

        W_conv.copyTo(expected_W_conv);
        b_conv.copyTo(expected_b_conv);
        W.copyTo(expected_W);
        b.copyTo(expected_b);

        gradW_conv  = model.getLayer("conv1").params().at("weight").grad();
        gradb_conv  = model.getLayer("conv1").params().at("bias").grad();
        gradW       = model.getLayer("linear2").params().at("weight").grad();
        gradb       = model.getLayer("linear2").params().at("bias").grad();

        expected_gradW_conv  = dw::Tensor{gradW_conv.shape()};
        expected_gradb_conv  = dw::Tensor{gradb_conv.shape()};
        expected_gradW       = dw::Tensor{gradW.shape()};
        expected_gradb       = dw::Tensor{gradb.shape()};

        dw::initializer::zeros(gradW_conv);
        dw::initializer::zeros(gradb_conv);
        dw::initializer::zeros(gradW);

        gradW_conv.copyTo(expected_gradW_conv);
        gradb_conv.copyTo(expected_gradb_conv);
        gradW.copyTo(expected_gradW);
        gradb.copyTo(expected_gradb);
    }

    void forward_reference(const dw::Tensor input, dw::Tensor& output) {
        dw::reference::CPUConvolution2DForward(input, expected_W_conv, expected_b_conv, conv_out,
                                               kernel_conv, padding_conv, stride_conv);

        dw::reference::CPULinearForward(conv_out.data(), expected_W.data(), output.data(),
                                        batch_size, mid_features, out_features);
    }

    void backward_reference(const dw::Tensor& input, const dw::Tensor& output, const dw::Tensor& grad_output, dw::Tensor& grad_input) {
        dw::reference::CPULinearBackward(conv_out.data(), expected_W.data(), grad_output.data(),
                                         expected_gradW.data(), conv_gradout.data(),
                                         batch_size, mid_features, out_features);
        dw::reference::CPULinearBiasBackward(grad_output.data(), expected_gradb.data(),
                                             batch_size, out_features);

        dw::reference::CPUConvolution2DBackward(input, conv_gradout, expected_W_conv, expected_b_conv,
                                                expected_gradW_conv,  expected_gradb_conv, grad_input,
                                                kernel_conv, padding_conv, stride_conv);
    }

    void validate() {
        // Validate output
        dw::testutils::AssertTensorEqual(output, expected);
        // Validate params
        dw::testutils::AssertTensorEqual(W_conv, expected_W_conv);
        dw::testutils::AssertTensorEqual(b_conv, expected_b_conv);
        dw::testutils::AssertTensorEqual(W, expected_W);
        dw::testutils::AssertTensorEqual(b, expected_b);
        // Validate gradients
        dw::testutils::AssertTensorEqual(gradW_conv, expected_gradW_conv);
        dw::testutils::AssertTensorEqual(gradb_conv, expected_gradb_conv);
        dw::testutils::AssertTensorEqual(gradW, expected_gradW);
        dw::testutils::AssertTensorEqual(gradb, expected_gradb);
    }

    dw::Model buildModel() {
        dw::Placeholder in(dw::Shape{batch_size, in_channels, in_features, in_features});
        auto out = dw::Convolution(out_channels, kernel_conv, padding_conv, stride_conv, "conv1")(in);
        out = dw::Linear(out_features, "linear2")(out);
        return {in, out};
    }

    std::array<int, 2> kernel_conv{5, 5};
    std::array<int, 2> padding_conv{2, 2};
    std::array<int, 2> stride_conv{1, 1};

    int in_features  = 28;
    int in_channels  = 1;
    int out_channels = 4;
    int mid_features = 4 * 28 * 28;
    int out_features = 10;
    int batch_size   = 32;

    dw::Placeholder in;
    dw::Model       model;

    dw::Tensor conv_out {dw::Shape{batch_size, out_channels, in_features, in_features}};
    dw::Tensor conv_gradout {dw::Shape{batch_size, out_channels, in_features, in_features}};

    dw::Tensor W, b, W_conv, b_conv;
    dw::Tensor expected_W, expected_b, expected_W_conv, expected_b_conv;

    dw::Tensor gradW, gradb, gradW_conv, gradb_conv;
    dw::Tensor expected_gradW, expected_gradb, expected_gradW_conv, expected_gradb_conv;

    dw::Tensor output{model.outputs()[0].shape()};
    dw::Tensor expected{model.outputs()[0].shape()};
    dw::Tensor grad_output{output.shape()};
};

TEST_F(CPULinearAfterConvolutionModelTest, CPULinearForwardAndBackward) {
    // Init
    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);
    dw::Tensor grad_output(output.shape());
    dw::Tensor grad_input(input.shape());
    dw::initializer::uniform(grad_output);

    // Deepworks
    model.forward(input, output);
    model.backward(input, output, grad_output);

    // Reference
    forward_reference(input, expected);
    backward_reference(input, expected, grad_output, grad_input);

    // Assert
    validate();
}
