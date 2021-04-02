#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

struct MNISTModel: public ::testing::Test {
    MNISTModel()
        : in(dw::Shape{batch_size, in_features}),
          model(buildModel()) {
        model.compile();
        W0 = model.getLayer("linear0").params()[0].data();
        b0 = model.getLayer("linear0").params()[1].data();
        W1 = model.getLayer("linear2").params()[0].data();
        b1 = model.getLayer("linear2").params()[1].data();
    }

    // NB: in{batch_size, in_feautres} -> Linear0(mid_features) -> l0out{batch_size, mid_features}
    // -> ReLU1() -> r1out{batch_size, mid_features} -> Linear2(out_features) -> l2out{batch_size, out_features}
    // -> Softmax3() -> s3out{batch_size, out_features}

    void forward_reference(const dw::Tensor input, dw::Tensor& output) {
        dw::reference::CPULinearForward(input.data(), W0.data(), linear_out0.data(),
                                               batch_size, in_features, mid_features);
        dw::reference::CPULinearAddBias(b0.data(), linear_out0.data(), batch_size, mid_features);

        dw::reference::CPUReLUForward(linear_out0.data(), relu_out1.data(), relu_out1.total());

        dw::reference::CPULinearForward(relu_out1.data(), W1.data(), linear_out2.data(),
                                        batch_size, mid_features, out_features);
        dw::reference::CPULinearAddBias(b1.data(), linear_out2.data(), batch_size, out_features);

        deepworks::reference::CPUSoftmaxForward(linear_out2.data(), output.data(),
                                                linear_out2.shape()[0], linear_out2.shape()[1]);
    }

    void backward_reference(const dw::Tensor& input,
                            const dw::Tensor& output,
                            const dw::Tensor& grad_output) {
        dw::reference::CPUSoftmaxBackward(grad_output.data(), output.data(), linear2_gradout.data(),
                                          linear2_gradout.shape()[0], linear2_gradout.shape()[1]);

        dw::reference::CPULinearBackward(relu_out1.data(), W1.data(), linear2_gradout.data(),
                                         expected_gradW1.data(), relu1_gradout.data(),
                                         batch_size, mid_features, out_features);

        dw::reference::CPULinearBiasBackward(linear2_gradout.data(), expected_gradb1.data(),
                                             batch_size, out_features);

        dw::reference::CPUReLUBackward(linear_out0.data(), relu1_gradout.data(), linear0_gradout.data(),
                                       batch_size, mid_features);

        dw::reference::CPULinearBackward(input.data(), W0.data(), linear0_gradout.data(),
                                         expected_gradW0.data(), grad_input.data(),
                                         batch_size, in_features, mid_features);

        dw::reference::CPULinearBiasBackward(linear0_gradout.data(), expected_gradb0.data(),
                                             batch_size, mid_features);
    }

    dw::Model buildModel() {
        dw::Placeholder in(dw::Shape{batch_size, in_features});
        auto out = dw::Linear(mid_features, "linear0")(in);
        out = dw::ReLU("relu1")(out);
        out = dw::Linear(out_features, "linear2")(out);
        out = dw::Softmax("softmax3")(out);
        return {in, out};
    }

    int in_features  = 32*32;
    int mid_features = 100;
    int out_features = 10;
    int batch_size   = 8;

    dw::Placeholder in;
    dw::Model model;

    // NB: Intermediate tensors (Forward)
    dw::Tensor linear_out0    {dw::Shape{batch_size, mid_features}};
    dw::Tensor relu_out1      {dw::Shape{batch_size, mid_features}};
    dw::Tensor linear_out2    {dw::Shape{batch_size, out_features}};
    // NB: Intermediate tensors (Backward)
    dw::Tensor linear2_gradout{dw::Shape{batch_size, out_features}};
    dw::Tensor relu1_gradout  {dw::Shape{batch_size, mid_features}};
    dw::Tensor linear0_gradout{dw::Shape{batch_size, mid_features}};
    dw::Tensor grad_input     {dw::Shape{batch_size, in_features}};

    dw::Tensor expected_gradW0{dw::Shape{mid_features, in_features}},
               expected_gradb0{dw::Shape{mid_features}},
               expected_gradW1{dw::Shape{out_features, mid_features}},
               expected_gradb1{dw::Shape{out_features}};

    dw::Tensor W0, b0, W1, b1;
};

TEST_F(MNISTModel, Forward) {
    // Init
    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);
    dw::Tensor output(model.outputs()[0].shape());
    dw::Tensor expected(model.outputs()[0].shape());

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    dw::testutils::AssertTensorEqual(output, expected);
}

TEST_F(MNISTModel, Backward) {
    // Init
    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);
    dw::Tensor output(model.outputs()[0].shape());
    dw::Tensor ref_output(output.shape());
    dw::Tensor grad_output(output.shape());
    dw::initializer::uniform(grad_output);

    // Deepworks
    model.forward(input, output);
    model.backward(input, output, grad_output);

    // Reference
    forward_reference(input, ref_output);
    backward_reference(input, ref_output, grad_output);

    // Assert
    auto gradW1 = model.getLayer("linear2").params()[0].grad();
    auto gradb1 = model.getLayer("linear2").params()[1].grad();
    auto gradW0 = model.getLayer("linear0").params()[0].grad();
    auto gradb0 = model.getLayer("linear0").params()[1].grad();

    dw::testutils::AssertTensorEqual(gradW1, expected_gradW1);
    dw::testutils::AssertTensorEqual(gradb1, expected_gradb1);
    dw::testutils::AssertTensorEqual(gradW0, expected_gradW0);
    dw::testutils::AssertTensorEqual(gradb0, expected_gradb0);
}
