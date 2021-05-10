#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

#include <random>

namespace dw = deepworks;

namespace {

struct MNISTModel: public ::testing::Test {
    MNISTModel()
        : in(dw::Shape{batch_size, in_features}),
          model(buildModel()) {
        model.compile();
        W0    = model.getLayer("linear0").params().at("weight").data();
        b0    = model.getLayer("linear0").params().at("bias").data();
        gamma = model.getLayer("batchnorm1d").params().at("gamma").data();
        beta  = model.getLayer("batchnorm1d").params().at("beta").data();
        W1    = model.getLayer("linear2").params().at("weight").data();
        b1    = model.getLayer("linear2").params().at("bias").data();

        expected_params.emplace("linear0.weight"   , dw::Tensor{W0.shape()});
        expected_params.emplace("linear0.bias"     , dw::Tensor{b0.shape()});
        expected_params.emplace("batchnorm1d.gamma", dw::Tensor{gamma.shape()});
        expected_params.emplace("batchnorm1d.beta" , dw::Tensor{beta.shape()});
        expected_params.emplace("linear2.weight"   , dw::Tensor{W1.shape()});
        expected_params.emplace("linear2.bias"     , dw::Tensor{b1.shape()});

        // NB: To easy access on specific parameter in tests.
        expected_W0    = expected_params.at("linear0.weight").data();
        expected_b0    = expected_params.at("linear0.bias").data();
        expected_gamma = expected_params.at("batchnorm1d.gamma").data();
        expected_beta  = expected_params.at("batchnorm1d.beta").data();
        expected_W1    = expected_params.at("linear2.weight").data();
        expected_b1    = expected_params.at("linear2.bias").data();

        W0.copyTo(expected_W0);
        b0.copyTo(expected_b0);
        gamma.copyTo(expected_gamma);
        beta.copyTo(expected_beta);
        W1.copyTo(expected_W1);
        b1.copyTo(expected_b1);

        gradW0    = model.getLayer("linear0").params().at("weight").grad();
        gradb0    = model.getLayer("linear0").params().at("bias").grad();
        gradGamma = model.getLayer("batchnorm1d").params().at("gamma").grad();
        gradBeta  = model.getLayer("batchnorm1d").params().at("beta").grad();
        gradW1    = model.getLayer("linear2").params().at("weight").grad();
        gradb1    = model.getLayer("linear2").params().at("bias").grad();

        // NB: To easy access on specific parameter in tests.
        expected_gradW0    = expected_params.at("linear0.weight").grad();
        expected_gradb0    = expected_params.at("linear0.bias").grad();
        expected_gradGamma = expected_params.at("batchnorm1d.gamma").grad();
        expected_gradBeta  = expected_params.at("batchnorm1d.beta").grad();
        expected_gradW1    = expected_params.at("linear2.weight").grad();
        expected_gradb1    = expected_params.at("linear2.bias").grad();

        // NB: Not to compare trash against trash in tests
        dw::initializer::zeros(gradW0);
        dw::initializer::zeros(gradb0);
        dw::initializer::zeros(gradGamma);
        dw::initializer::zeros(gradBeta);
        dw::initializer::zeros(gradW1);
        dw::initializer::zeros(gradb1);

        gradW0.copyTo(expected_gradW0);
        gradb0.copyTo(expected_gradb0);
        gradGamma.copyTo(expected_gradGamma);
        gradBeta.copyTo(expected_gradBeta);
        gradW1.copyTo(expected_gradW1);
        gradb1.copyTo(expected_gradb1);

        // NB: Not to compare trash against trash in tests
        dw::initializer::zeros(grad_output);
        grad_output.copyTo(expected_grad_output);

        ref_input_centered = dw::Tensor::zeros({batch_size, mid_features});
        ref_std            = dw::Tensor::zeros({mid_features});
        ref_moving_mean    = dw::Tensor::zeros({mid_features});
        ref_moving_var     = dw::Tensor::zeros({mid_features});

        loss = 0.f;
        expected_loss = loss;
    }

    // NB: in{batch_size, in_feautres} -> Linear0(mid_features) -> l0out{batch_size, mid_features}
    // -> ReLU1() -> r1out{batch_size, mid_features}
    // -> BatchNorm2() -> b2out{batch_size, mid_features}
    // -> Linear3(out_features) -> l3out{batch_size, out_features}
    // -> Softmax4() -> s4out{batch_size, out_features}

    void forward_reference(const dw::Tensor input, dw::Tensor& output) {
        dw::reference::CPULinearForward(input.data(), expected_W0.data(), linear_out0.data(),
                                        batch_size, in_features, mid_features);
        dw::reference::CPULinearAddBias(expected_b0.data(), linear_out0.data(), batch_size, mid_features);

        dw::reference::CPUReLUForward(linear_out0.data(), relu_out1.data(), relu_out1.total());

        dw::reference::CPUBatchNorm1DForward(relu_out1, batch_norm_out2,
                                             ref_input_centered, ref_std,
                                             ref_moving_mean, ref_moving_var,
                                             train, epsilon, alpha,
                                             expected_gamma, expected_beta);

        dw::reference::CPULinearForward(batch_norm_out2.data(), expected_W1.data(), linear_out3.data(),
                                        batch_size, mid_features, out_features);
        dw::reference::CPULinearAddBias(expected_b1.data(), linear_out3.data(), batch_size, out_features);

        dw::reference::CPUSoftmaxForward(linear_out3.data(), output.data(),
                                         linear_out3.shape()[0], linear_out3.shape()[1]);
    }

    void backward_reference(const dw::Tensor& input,
                            const dw::Tensor& output,
                            const dw::Tensor& grad_output) {
        dw::reference::CPUSoftmaxBackward(grad_output.data(), output.data(), linear3_gradout.data(),
                                          linear3_gradout.shape()[0], linear3_gradout.shape()[1]);

        dw::reference::CPULinearBackward(batch_norm_out2.data(), expected_W1.data(), linear3_gradout.data(),
                                         expected_gradW1.data(), batch_norm2_gradout.data(),
                                         batch_size, mid_features, out_features);

        dw::reference::CPULinearBiasBackward(linear3_gradout.data(), expected_gradb1.data(),
                                             batch_size, out_features);

        dw::reference::CPUBatchNorm1DBackward(ref_input_centered, ref_std, batch_norm2_gradout,
                                              relu1_gradout, expected_gamma, expected_gradGamma, expected_gradBeta);

        dw::reference::CPUReLUBackward(linear_out0.data(), relu1_gradout.data(), linear0_gradout.data(),
                                       linear_out0.total());
        dw::reference::CPULinearBackward(input.data(), expected_W0.data(), linear0_gradout.data(),
                                         expected_gradW0.data(), grad_input.data(),
                                         batch_size, in_features, mid_features);
        dw::reference::CPULinearBiasBackward(linear0_gradout.data(), expected_gradb0.data(),
                                             batch_size, mid_features);
    }

    void validate() {
        // Validate output
        dw::testutils::AssertTensorEqual(output, expected);
        // Validate grad outputs
        dw::testutils::AssertTensorEqual(grad_output, expected_grad_output);
        // Validate params
        dw::testutils::AssertTensorEqual(W1, expected_W1);
        dw::testutils::AssertTensorEqual(b1, expected_b1);
        dw::testutils::AssertTensorEqual(gamma, expected_gamma);
        dw::testutils::AssertTensorEqual(beta, expected_beta);
        dw::testutils::AssertTensorEqual(W0, expected_W0);
        dw::testutils::AssertTensorEqual(b0, expected_b0);
        // Validate gradients
        dw::testutils::AssertTensorEqual(gradW1, expected_gradW1);
        dw::testutils::AssertTensorEqual(gradb1, expected_gradb1);
        dw::testutils::AssertTensorEqual(gradGamma, expected_gradGamma);
        dw::testutils::AssertTensorEqual(gradBeta, expected_gradBeta);
        dw::testutils::AssertTensorEqual(gradW0, expected_gradW0);
        dw::testutils::AssertTensorEqual(gradb0, expected_gradb0);
        // Validate loss
        dw::testutils::AssertEqual(loss, expected_loss);
    }

    dw::Model buildModel() {
        dw::Placeholder in(dw::Shape{batch_size, in_features});

        auto out = dw::Linear(mid_features, "linear0")(in);
        out = dw::ReLU("relu1")(out);
        out = dw::BatchNorm1D(epsilon, alpha, "batchnorm1d")(out);
        out = dw::Linear(out_features, "linear2")(out);
        out = dw::Softmax("softmax3")(out);
        return {in, out};
    }

    bool train = true;

    int in_features  = 28*28;
    int mid_features = 100;
    int out_features = 10;
    int batch_size   = 8;

    float epsilon = 0.001;
    float alpha   = 0.05;

    dw::Placeholder in;
    dw::Model model;

    // NB: Helper tensors for BatchNorm
    dw::Tensor ref_input_centered;
    dw::Tensor ref_std;
    dw::Tensor ref_moving_mean;
    dw::Tensor ref_moving_var;
    // NB: Intermediate tensors (Forward)
    dw::Tensor linear_out0    {dw::Shape{batch_size, mid_features}};
    dw::Tensor relu_out1      {dw::Shape{batch_size, mid_features}};
    dw::Tensor batch_norm_out2{dw::Shape{batch_size, mid_features}};
    dw::Tensor linear_out3    {dw::Shape{batch_size, out_features}};
    // NB: Intermediate tensors (Backward)
    dw::Tensor linear3_gradout    {dw::Shape{batch_size, out_features}};
    dw::Tensor batch_norm2_gradout{dw::Shape{batch_size, mid_features}};
    dw::Tensor relu1_gradout      {dw::Shape{batch_size, mid_features}};
    dw::Tensor linear0_gradout    {dw::Shape{batch_size, mid_features}};
    dw::Tensor grad_input         {dw::Shape{batch_size, in_features}};

    dw::Tensor output{model.outputs()[0].shape()};
    dw::Tensor expected{output.shape()};
    dw::Tensor grad_output{output.shape()};
    dw::Tensor expected_grad_output{grad_output.shape()};

    dw::Tensor expected_W0, expected_b0, expected_gamma, expected_beta, expected_W1, expected_b1;
    dw::Tensor expected_gradW0{dw::Shape{mid_features, in_features}},
               expected_gradb0{dw::Shape{mid_features}},
               expected_gradGamma{dw::Shape{mid_features}},
               expected_gradBeta{dw::Shape{mid_features}},
               expected_gradW1{dw::Shape{out_features, mid_features}},
               expected_gradb1{dw::Shape{out_features}};

    dw::Tensor W0, b0, gamma, beta, W1, b1;
    dw::Tensor gradW0, gradb0, gradGamma, gradBeta, gradW1, gradb1;

    dw::ParamMap expected_params;

    float loss, expected_loss;
};

struct Dataset : public dw::IDataset {
public:
    struct Data {
        dw::Tensor X;
        dw::Tensor y;
    };

    Dataset(int size) : m_data(size) {
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> dist(0, 9);

        for (auto&& d : m_data) {
            d.X.allocate(dw::Shape{28 * 28});
            dw::initializer::uniform(d.X);

            d.y.allocate(dw::Shape{1});
            d.y.data()[0] = static_cast<float>(dist(gen));
        }
    }

    size_t                 size()  override { return m_data.size();                    }
    dw::IDataset::OutShape shape() override { return {dw::Shape{28*28}, dw::Shape{1}}; }

    void pull(int idx, dw::Tensor& X, dw::Tensor& y) override {
        m_data[idx].X.copyTo(X);
        m_data[idx].y.copyTo(y);
    }

private:
    std::vector<Data> m_data;
};

} // anonymous namespace

TEST_F(MNISTModel, Forward) {
    // Init
    auto input = dw::Tensor::uniform(in.shape());

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    validate();
}

TEST_F(MNISTModel, ForwardTrainFalse) {
    // Init
    auto input = dw::Tensor::uniform(in.shape());

    model.train(false);
    train = false;

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    validate();
}

TEST_F(MNISTModel, Backward) {
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
    validate();
}

TEST_F(MNISTModel, TrainLoopSmoke) {
    size_t num_imgs = 100u;
    int num_epochs  = 10;
    auto loader     = dw::DataLoader(std::make_shared<Dataset>(num_imgs), batch_size, false);

    dw::Tensor X, y;

    // Deepworks train loop
    dw::loss::CrossEntropyLoss criterion;
    dw::optimizer::SGD opt(model.params(), 0.01);
    for (int i = 0; i < num_epochs; ++i) {
        while (loader.pull(X, y)) {
            model.forward(X, output);
            loss = criterion.forward(output, y);
            criterion.backward(output, y, grad_output);
            model.backward(X, output, grad_output);
            opt.step();
        }
    }

    // Reference train loop
    for (int i = 0; i < num_epochs; ++i) {
        while (loader.pull(X, y)) {
            forward_reference(X, expected);
            expected_loss = dw::reference::CPUCrossEntropyLossForward(expected, y);
            dw::reference::CPUCrossEntropyLossBackward(expected, y, expected_grad_output);
            backward_reference(X, expected, expected_grad_output);
            dw::reference::SGDStep(expected_params, opt.get_lr());
        }
    }

    // Assert
    validate();
}

TEST_F(MNISTModel, SaveLoadParams) {
    // Init
    auto input       = dw::Tensor::uniform(in.shape());
    auto grad_output = dw::Tensor::uniform(output.shape());

    // Deepworks
    model.forward(input, output);
    model.backward(input, output, grad_output);

    // Save original model
    dw::save_state(model.state(), "state.bin");

    // Load to another model
    auto another = buildModel();
    dw::load_state(another.state(), "state.bin");

    // Validate
    for (const auto&[name, param] : model.state()) {
        dw::testutils::AssertTensorEqual(param, another.state().at(name));
    }
}

TEST_F(MNISTModel, SaveCfgToDot) {
    EXPECT_NO_THROW(dw::save_dot(model.cfg(), "model.dot"));
}

TEST_F(MNISTModel, SaveLoadConfig) {
    dw::save_cfg(model.cfg(), "cfg.bin");

    auto another = dw::Model::Build(dw::load_cfg("cfg.bin"));

    ASSERT_EQ(model.inputs().size(), another.inputs().size());
    for (int i = 0; i < model.inputs().size(); ++i) {
        EXPECT_EQ(model.inputs()[i].shape(), another.inputs()[i].shape());
    }

    ASSERT_EQ(model.outputs().size(), another.outputs().size());
    for (int i = 0; i < model.outputs().size(); ++i) {
        EXPECT_EQ(model.outputs()[i].shape(), another.outputs()[i].shape());
    }

    ASSERT_EQ(model.params().size(), another.params().size());
    for (const auto& [name, param] : model.params()) {
        EXPECT_EQ(param.data().shape(), another.params().at(name).data().shape());
        EXPECT_EQ(param.grad().shape(), another.params().at(name).grad().shape());
    }

    ASSERT_EQ(model.layers().size(), another.layers().size());
    for (int i = 0; i < model.layers().size(); ++i) {
        const auto& expected_layer = model.layers()[i];
        const auto& actual_layer   = another.layers()[i];
        EXPECT_EQ(expected_layer.name(), actual_layer.name());
        EXPECT_EQ(expected_layer.type(), actual_layer.type());

        ASSERT_EQ(expected_layer.inputs().size(), actual_layer.inputs().size());
        for (int j = 0; j < expected_layer.inputs().size(); ++j) {
            EXPECT_EQ(expected_layer.inputs()[j].shape(), actual_layer.inputs()[j].shape());
        }

        ASSERT_EQ(expected_layer.outputs().size(), actual_layer.outputs().size());
        for (int j = 0; j < expected_layer.outputs().size(); ++j) {
            EXPECT_EQ(expected_layer.outputs()[j].shape(), actual_layer.outputs()[j].shape());
        }

        ASSERT_EQ(expected_layer.params().size(), actual_layer.params().size());
        for (const auto& [name, param] : expected_layer.params()) {
            EXPECT_EQ(param.data().shape(), actual_layer.params().at(name).data().shape());
            EXPECT_EQ(param.grad().shape(), actual_layer.params().at(name).grad().shape());
        }

        ASSERT_EQ(expected_layer.buffers().size(), actual_layer.buffers().size());
        for (const auto& [name, buffer] : expected_layer.buffers()) {
            EXPECT_EQ(buffer.shape(), actual_layer.buffers().at(name).shape());
        }
    }
}

TEST_F(MNISTModel, SaveLoadFullModel) {
    // Init
    auto input       = dw::Tensor::uniform(in.shape());
    auto grad_output = dw::Tensor::uniform(output.shape());

    // Deepworks
    model.forward(input, output);
    model.backward(input, output, grad_output);

    // Save original model
    dw::save(model, "model.bin");

    // Load to another model
    auto another = dw::load("model.bin");

    // Validate
    for (const auto&[name, param] : model.state()) {
        dw::testutils::AssertTensorEqual(param, another.state().at(name));
    }
}

namespace {

struct CIFAR10Model : public ::testing::Test {
    CIFAR10Model()
            : in(dw::Shape{batch_size, in_channels, image_size, image_size}),
              model(buildModel()) {
        model.compile();

        Wconv0 = model.getLayer("conv0").params().at("weight").data();
        bconv0 = model.getLayer("conv0").params().at("bias").data();
        Wconv = model.getLayer("conv1").params().at("weight").data();
        bconv = model.getLayer("conv1").params().at("bias").data();
        W     = model.getLayer("linear4").params().at("weight").data();
        b     = model.getLayer("linear4").params().at("bias").data();

        expected_params.emplace("conv0.weight", dw::Tensor{Wconv0.shape()});
        expected_params.emplace("conv0.bias", dw::Tensor{bconv0.shape()});
        expected_params.emplace("conv.weight", dw::Tensor{Wconv.shape()});
        expected_params.emplace("conv.bias", dw::Tensor{bconv.shape()});
        expected_params.emplace("linear.weight", dw::Tensor{W.shape()});
        expected_params.emplace("linear.bias", dw::Tensor{b.shape()});

        // NB: To easy access on specific parameter in tests.
        expected_Wconv0 = expected_params.at("conv0.weight").data();
        expected_bconv0 = expected_params.at("conv0.bias").data();
        expected_Wconv = expected_params.at("conv.weight").data();
        expected_bconv = expected_params.at("conv.bias").data();
        expected_W     = expected_params.at("linear.weight").data();
        expected_b     = expected_params.at("linear.bias").data();

        Wconv0.copyTo(expected_Wconv0);
        bconv0.copyTo(expected_bconv0);
        Wconv.copyTo(expected_Wconv);
        bconv.copyTo(expected_bconv);
        W.copyTo(expected_W);
        b.copyTo(expected_b);

        gradWconv0 = model.getLayer("conv0").params().at("weight").grad();
        gradbconv0 = model.getLayer("conv0").params().at("bias").grad();
        gradWconv = model.getLayer("conv1").params().at("weight").grad();
        gradbconv = model.getLayer("conv1").params().at("bias").grad();
        gradW     = model.getLayer("linear4").params().at("weight").grad();
        gradb     = model.getLayer("linear4").params().at("bias").grad();

        // NB: To easy access on specific parameter in tests.
        expected_gradWconv0 = expected_params.at("conv0.weight").grad();
        expected_gradbconv0 = expected_params.at("conv0.bias").grad();
        expected_gradWconv = expected_params.at("conv.weight").grad();
        expected_gradbconv = expected_params.at("conv.bias").grad();
        expected_gradW     = expected_params.at("linear.weight").grad();
        expected_gradb     = expected_params.at("linear.bias").grad();

        // NB: Not to compare trash against trash in tests
        dw::initializer::zeros(gradWconv0);
        dw::initializer::zeros(gradbconv0);
        dw::initializer::zeros(gradWconv);
        dw::initializer::zeros(gradbconv);
        dw::initializer::zeros(gradW);
        dw::initializer::zeros(gradb);

        gradWconv0.copyTo(expected_gradWconv0);
        gradbconv0.copyTo(expected_gradbconv0);
        gradWconv.copyTo(expected_gradWconv);
        gradbconv.copyTo(expected_gradbconv);
        gradW.copyTo(expected_gradW);
        gradb.copyTo(expected_gradb);

        // NB: Not to compare trash against trash in tests
        dw::initializer::zeros(grad_output);
        grad_output.copyTo(expected_grad_output);

        loss          = 0.f;
        expected_loss = loss;
    }

    // NB: in{batch_size, in_channels, image_size, image_size}
    // -> Convolution0(out_channels, <params>_conv) -> l0out{batch_size, out_channels, image_size, image_size}
    // -> Convolution1(out_channels, <params>_conv) -> l1out{batch_size, out_channels, image_size, image_size}
    // -> MaxPooling2(<params>_pool) -> mp2out{batch_size, out_channels, image_size/2, image_size/2}
    // -> ReLU3() -> r3out{batch_size, out_channels, image_size/2, image_size/2}
    // -> Linear4(out_features) -> l4out{batch_size, out_features}
    // -> Softmax5() -> s5out{batch_size, out_features}

    void forward_reference(const dw::Tensor input, dw::Tensor& output) {
        dw::reference::CPUConvolution2DForward(input, expected_Wconv0, expected_bconv0, conv_out0,
                                               kernel_conv, padding_conv, stride_conv);

        dw::reference::CPUConvolution2DForward(conv_out0, expected_Wconv, expected_bconv, conv_out1,
                                               kernel_conv, padding_conv, stride_conv);

        dw::reference::CPUMaxPooling2DForward(conv_out1, maxpool_out2, kernel_pool, padding_pool, stride_pool);

        dw::reference::CPUReLUForward(maxpool_out2.data(), relu_out3.data(), relu_out3.total());

        dw::reference::CPULinearForward(relu_out3.data(), expected_W.data(), linear_out4.data(),
                                        batch_size, mid_features, out_features);
        dw::reference::CPULinearAddBias(expected_b.data(), linear_out4.data(), batch_size, out_features);

        dw::reference::CPUSoftmaxForward(linear_out4.data(), output.data(),
                                         linear_out4.shape()[0], linear_out4.shape()[1]);
    }

    void backward_reference(const dw::Tensor& input,
                            const dw::Tensor& output,
                            const dw::Tensor& grad_output) {
        dw::reference::CPUSoftmaxBackward(grad_output.data(), output.data(), linear4_gradout.data(),
                                          linear4_gradout.shape()[0], linear4_gradout.shape()[1]);

        dw::reference::CPULinearBackward(relu_out3.data(), expected_W.data(), linear4_gradout.data(),
                                         expected_gradW.data(), relu3_gradout.data(),
                                         batch_size, mid_features, out_features);
        dw::reference::CPULinearBiasBackward(linear4_gradout.data(), expected_gradb.data(),
                                             batch_size, out_features);

        dw::reference::CPUReLUBackward(maxpool_out2.data(), relu3_gradout.data(), maxpool2_gradout.data(),
                                       maxpool_out2.total());

        dw::reference::CPUMaxPooling2DBackward(conv_out1, maxpool2_gradout, conv1_gradout,
                                               kernel_pool, padding_pool, stride_pool);

        dw::reference::CPUConvolution2DBackward(conv_out0, conv1_gradout, expected_Wconv, expected_bconv,
                                                expected_gradWconv, expected_gradbconv, conv0_gradout,
                                                kernel_conv, padding_conv, stride_conv);

        dw::reference::CPUConvolution2DBackward(input, conv0_gradout, expected_Wconv0, expected_bconv0,
                                                expected_gradWconv0, expected_gradbconv0, grad_input,
                                                kernel_conv, padding_conv, stride_conv);
    }

    void validate() {
        // Validate output
        dw::testutils::AssertTensorEqual(output, expected);
        // Validate grad outputs
        dw::testutils::AssertTensorEqual(grad_output, expected_grad_output);
        // Validate params
        dw::testutils::AssertTensorEqual(Wconv0, expected_Wconv0);
        dw::testutils::AssertTensorEqual(bconv0, expected_bconv0);
        dw::testutils::AssertTensorEqual(Wconv, expected_Wconv);
        dw::testutils::AssertTensorEqual(bconv, expected_bconv);
        dw::testutils::AssertTensorEqual(W, expected_W);
        dw::testutils::AssertTensorEqual(b, expected_b);
        // Validate gradients
        dw::testutils::AssertTensorEqual(gradWconv0, expected_gradWconv0);
        dw::testutils::AssertTensorEqual(gradbconv0, expected_gradbconv0);
        dw::testutils::AssertTensorEqual(gradWconv, expected_gradWconv);
        dw::testutils::AssertTensorEqual(gradbconv, expected_gradbconv);
        dw::testutils::AssertTensorEqual(gradW, expected_gradW);
        dw::testutils::AssertTensorEqual(gradb, expected_gradb);
        // Validate loss
        dw::testutils::AssertEqual(loss, expected_loss);
    }

    dw::Model buildModel() {
        dw::Placeholder in(dw::Shape{batch_size, in_channels, image_size, image_size});

        auto out = dw::Convolution(out_channels, kernel_conv, padding_conv, stride_conv, "conv0")(in);
        out = dw::Convolution(out_channels, kernel_conv, padding_conv, stride_conv, "conv1")(out);
        out = dw::MaxPooling(kernel_pool, padding_pool, stride_pool, "pool2")(out);
        out = dw::ReLU("relu3")(out);
        out = dw::Linear(out_features, "linear4")(out);
        out = dw::Softmax("softmax5")(out);
        return {in, out};
    }

    bool train = true;

    int batch_size   = 64;
    int in_channels  = 3;
    int out_channels = 4;
    int image_size   = 32;
    int pool_features = 16;
    int mid_features = out_channels * pool_features * pool_features;
    int out_features = 10;

    std::array<int, 2> kernel_conv{5, 5};
    std::array<int, 2> padding_conv{2, 2};
    std::array<int, 2> stride_conv{1, 1};

    std::array<int, 2> kernel_pool{2, 2};
    std::array<int, 2> padding_pool{0, 0};
    std::array<int, 2> stride_pool{2, 2};

    dw::Placeholder in;
    dw::Model       model;

    // NB: Intermediate tensors (Forward)
    dw::Tensor conv_out0{dw::Shape{batch_size, out_channels, image_size, image_size}};
    dw::Tensor conv_out1{dw::Shape{batch_size, out_channels, image_size, image_size}};
    dw::Tensor maxpool_out2{dw::Shape{batch_size, out_channels, pool_features, pool_features}};
    dw::Tensor relu_out3{dw::Shape{batch_size, out_channels, pool_features, pool_features}};
    dw::Tensor linear_out4{dw::Shape{batch_size, out_features}};
    // NB: Intermediate tensors (Backward)
    dw::Tensor linear4_gradout{dw::Shape{batch_size, out_features}};
    dw::Tensor relu3_gradout{dw::Shape{batch_size, out_channels, pool_features, pool_features}};
    dw::Tensor maxpool2_gradout{dw::Shape{batch_size, out_channels, pool_features, pool_features}};
    dw::Tensor conv1_gradout{dw::Shape{batch_size, out_channels, image_size, image_size}};
    dw::Tensor conv0_gradout{dw::Shape{batch_size, out_channels, image_size, image_size}};
    dw::Tensor grad_input{dw::Shape{batch_size, in_channels, image_size, image_size}};

    dw::Tensor output{model.outputs()[0].shape()};
    dw::Tensor expected{output.shape()};
    dw::Tensor grad_output{output.shape()};
    dw::Tensor expected_grad_output{grad_output.shape()};

    dw::Tensor Wconv0, bconv0, Wconv, bconv, W, b;
    dw::Tensor gradWconv0, gradbconv0, gradWconv, gradbconv, gradW, gradb;

    dw::Tensor expected_Wconv0, expected_bconv0, expected_Wconv, expected_bconv, expected_W, expected_b;
    dw::Tensor expected_gradWconv0, expected_gradbconv0, expected_gradWconv, expected_gradbconv, expected_gradW, expected_gradb;

    dw::ParamMap expected_params;

    float loss, expected_loss;
};


struct CIFAR10Dataset : public dw::IDataset {
public:
    struct Data {
        dw::Tensor X;
        dw::Tensor y;
    };

    CIFAR10Dataset(int size) : m_data(size) {
        std::mt19937                    gen(std::random_device{}());
        std::uniform_int_distribution<> dist(0, 9);

        for (auto&& d : m_data) {
            d.X.allocate(dw::Shape{3, 32, 32});
            dw::initializer::uniform(d.X);

            d.y.allocate(dw::Shape{1});
            d.y.data()[0] = static_cast<float>(dist(gen));
        }
    }

    size_t size() override { return m_data.size(); }

    dw::IDataset::OutShape shape() override { return {dw::Shape{3, 32, 32}, dw::Shape{1}}; }

    void pull(int idx, dw::Tensor& X, dw::Tensor& y) override {
        m_data[idx].X.copyTo(X);
        m_data[idx].y.copyTo(y);
    }

private:
    std::vector<Data> m_data;
};
}

TEST_F(CIFAR10Model, Forward) {
    // Init
    auto input = dw::Tensor::uniform(in.shape());

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    validate();
}

TEST_F(CIFAR10Model, ForwardTrainFalse) {
    // Init
    auto input = dw::Tensor::uniform(in.shape());

    train = false;
    model.train(train);

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    validate();
}

TEST_F(CIFAR10Model, Backward) {
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
    validate();
}

TEST_F(CIFAR10Model, TrainLoopSmoke) {
    size_t num_imgs = 100u;
    int num_epochs  = 10;
    auto loader     = dw::DataLoader(std::make_shared<CIFAR10Dataset>(num_imgs), batch_size, false);

    dw::Tensor X, y;

    // Deepworks train loop
    dw::loss::CrossEntropyLoss criterion;
    dw::optimizer::SGD opt(model.params(), 0.01);
    for (int i = 0; i < num_epochs; ++i) {
        while (loader.pull(X, y)) {
            model.forward(X, output);
            loss = criterion.forward(output, y);
            criterion.backward(output, y, grad_output);
            model.backward(X, output, grad_output);
            opt.step();
        }
    }

    // Reference train loop
    for (int i = 0; i < num_epochs; ++i) {
        while (loader.pull(X, y)) {
            forward_reference(X, expected);
            expected_loss = dw::reference::CPUCrossEntropyLossForward(expected, y);
            dw::reference::CPUCrossEntropyLossBackward(expected, y, expected_grad_output);
            backward_reference(X, expected, expected_grad_output);
            dw::reference::SGDStep(expected_params, opt.get_lr());
        }
    }

    // Assert
    validate();
}

struct BatchNorm2dHelper {
    BatchNorm2dHelper(float a, float e)
        : alpha(a), epsilon(e), layer(epsilon, alpha) {
    }

    dw::Placeholder operator()(dw::Placeholder in) {
        auto out = layer(in);

        const auto& params = layer.m_info.params();
        gamma      = params.at("gamma").data();
        beta       = params.at("beta").data();
        gamma_grad = params.at("gamma").grad();
        beta_grad  = params.at("beta").grad();

        expected_gamma.allocate(gamma.shape());
        expected_beta.allocate(beta.shape());
        expected_gamma_grad.allocate(gamma_grad.shape());
        expected_beta_grad.allocate(beta_grad.shape());

        gamma.copyTo(expected_gamma);
        beta.copyTo(expected_beta);
        gamma_grad.copyTo(expected_gamma_grad);
        gamma_grad.copyTo(expected_beta_grad);

        const auto& in_shape = in.shape();
        int N = in_shape[0];
        int C = in_shape[1];
        int H = in_shape[2];
        int W = in_shape[3];

        moving_mean = dw::Tensor::zeros({N * H * W});
        moving_var  = dw::Tensor::zeros({N * H * W});
        input_centered.allocate(in.shape());
        std.allocate({gamma.shape()[0]});

        _output.allocate(out.shape());
        _grad_input.allocate(in.shape());

        return out;
    }

    void forward(const dw::Tensor& input, dw::Tensor& output) {
        dw::reference::CPUBatchNorm2DForward(input,
                                             output,
                                             expected_gamma,
                                             expected_beta,
                                             moving_mean,
                                             moving_var,
                                             input_centered,
                                             std,
                                             true,
                                             alpha,
                                             epsilon);
    }

    void backward(const dw::Tensor& grad_output,
                        dw::Tensor& grad_input) {
         dw::reference::CPUBatchNorm2DBackward(input_centered,
                                               std,
                                               grad_output,
                                               grad_input,
                                               expected_gamma,
                                               expected_gamma_grad,
                                               expected_beta_grad);
    }

    void validate() {
        dw::testutils::AssertTensorEqual(expected_gamma, gamma);
        dw::testutils::AssertTensorEqual(expected_beta, beta);
        dw::testutils::AssertTensorEqual(expected_gamma_grad, gamma_grad);
        dw::testutils::AssertTensorEqual(expected_beta_grad, beta_grad);
    }

    dw::Tensor gamma, beta, gamma_grad, beta_grad;
    dw::Tensor moving_mean, moving_var, input_centered, std;

    dw::Tensor expected_beta,
               expected_gamma,
               expected_gamma_grad,
               expected_beta_grad;

    dw::Tensor _output;
    dw::Tensor _grad_input;

    float alpha, epsilon;

    dw::BatchNorm2D layer;
};

struct ConvolutionHelper {
    ConvolutionHelper(int _out_channels,
                      const std::array<int, 2>& _kernel,
                      const std::array<int, 2>& _padding,
                      const std::array<int, 2>& _stride)
        : out_channels(_out_channels),
          kernel(_kernel),
          padding(_padding),
          stride(_stride),
          layer(out_channels, kernel, padding, stride) {
    }

    dw::Placeholder operator()(dw::Placeholder in) {
        auto out = layer(in);

        const auto& params = layer.m_info.params();
        weight      = params.at("weight").data();
        bias        = params.at("bias").data();
        weight_grad = params.at("weight").grad();
        bias_grad   = params.at("bias").grad();

        expected_weight.allocate(weight.shape());
        expected_bias.allocate(bias.shape());
        expected_weight_grad.allocate(weight_grad.shape());
        expected_bias_grad.allocate(bias_grad.shape());

        weight.copyTo(expected_weight);
        bias.copyTo(expected_bias);
        weight_grad.copyTo(expected_weight_grad);
        bias_grad.copyTo(expected_bias_grad);

        _output.allocate(out.shape());
        _grad_input.allocate(in.shape());

        return out;
    }

    void forward(const dw::Tensor& input, dw::Tensor& output) {
        dw::reference::CPUConvolution2DForward(input, expected_weight, expected_bias,
                                               output, kernel, padding, stride);
    }

    void backward(const dw::Tensor& input,
                  const dw::Tensor& grad_output,
                        dw::Tensor& grad_input) {
        dw::reference::CPUConvolution2DBackward(input, grad_output, expected_weight, expected_bias,
                                                expected_weight_grad, expected_bias_grad, grad_input,
                                                kernel, padding, stride);
    }

    void validate() {
        dw::testutils::AssertTensorEqual(expected_weight, weight);
        dw::testutils::AssertTensorEqual(expected_bias, bias);
        dw::testutils::AssertTensorEqual(expected_weight_grad, weight_grad);
        dw::testutils::AssertTensorEqual(expected_bias_grad, bias_grad);
    }

    int out_channels;
    std::array<int, 2> kernel;
    std::array<int, 2> padding;
    std::array<int, 2> stride;

    dw::Convolution layer;

    dw::Tensor weight, bias, weight_grad, bias_grad;
    dw::Tensor expected_weight,
               expected_bias,
               expected_weight_grad,
               expected_bias_grad;

    dw::Tensor _output;
    dw::Tensor _grad_input;
};

struct ReLUHelper {
    dw::Placeholder operator()(dw::Placeholder in) {
        auto out = layer(in);
        _output.allocate(out.shape());
        _grad_input.allocate(in.shape());
        return out;
    }

    void forward(const dw::Tensor& input, dw::Tensor& output) {
        dw::reference::CPUReLUForward(input.data(), output.data(), input.total());

    }

    void backward(const dw::Tensor& output,
                  const dw::Tensor& grad_output,
                        dw::Tensor& grad_input) {
        dw::reference::CPUReLUBackward(output.data(), grad_output.data(),
                                       grad_input.data(), output.total());
    }

    dw::ReLU layer;

    dw::Tensor _output;
    dw::Tensor _grad_input;
};

struct AddHelper {
    dw::Placeholder operator()(dw::Placeholder in0, dw::Placeholder in1) {
        auto out = layer(in0, in1);
        _output.allocate(out.shape());
        _grad_input0.allocate(in0.shape());
        _grad_input1.allocate(in1.shape());
        return out;
    }

    void forward(const dw::Tensor& input0,
                 const dw::Tensor& input1,
                 dw::Tensor& output) {
        dw::reference::CPUAddForward(input0, input1, output);
    }

    void backward(const dw::Tensor& grad_output,
                        dw::Tensor& grad_input0,
                        dw::Tensor& grad_input1) {
        dw::reference::CPUAddBackward(grad_output, grad_input0, grad_input1);
    }

    dw::Add layer;
    dw::Tensor _output;
    dw::Tensor _grad_input0;
    dw::Tensor _grad_input1;
};

struct GlobalAVGPoolHelper {
    dw::Placeholder operator()(dw::Placeholder in) {
        auto out = layer(in);
        _output.allocate(out.shape());
        _grad_input.allocate(in.shape());
        return out;
    }

    void forward(const dw::Tensor& input, dw::Tensor& output) {
        dw::reference::CPUGlobalAvgPoolingForward(input, output);

    }

    void backward(const dw::Tensor& grad_output,
                        dw::Tensor& grad_input) {
        dw::reference::CPUGlobalAvgPoolingBackward(grad_output, grad_input);
    }

    dw::GlobalAvgPooling layer;
    dw::Tensor _output;
    dw::Tensor _grad_input;
};

struct LinearHelper {
    LinearHelper(int _batch_size, int _in_features, int _out_features)
        : batch_size(_batch_size),
          in_features(_in_features),
          out_features(_out_features),
          layer(out_features) {
    }

    dw::Placeholder operator()(dw::Placeholder in) {
        auto out = layer(in);

        const auto& params = layer.m_info.params();
        weight      = params.at("weight").data();
        bias        = params.at("bias").data();
        weight_grad = params.at("weight").grad();
        bias_grad   = params.at("bias").grad();

        expected_weight.allocate(weight.shape());
        expected_bias.allocate(bias.shape());
        expected_weight_grad.allocate(weight_grad.shape());
        expected_bias_grad.allocate(bias_grad.shape());

        weight.copyTo(expected_weight);
        bias.copyTo(expected_bias);
        weight_grad.copyTo(expected_weight_grad);
        bias_grad.copyTo(expected_bias_grad);

        _output.allocate(out.shape());
        _grad_input.allocate(in.shape());

        return out;
    }

    void forward(const dw::Tensor& input, dw::Tensor& output) {
        dw::reference::CPULinearForward(input.data(), expected_weight.data(), output.data(),
                                        batch_size, in_features, out_features);

        dw::reference::CPULinearAddBias(expected_bias.data(), output.data(),
                                        batch_size, out_features);
    }

    void backward(const dw::Tensor& input,
                  const dw::Tensor& grad_output,
                        dw::Tensor& grad_input) {
        dw::reference::CPULinearBackward(input.data(), expected_weight.data(), grad_output.data(),
                                         expected_weight_grad.data(), grad_input.data(), batch_size,
                                         in_features, out_features);
        dw::reference::CPULinearBiasBackward(grad_output.data(), expected_bias_grad.data(), batch_size, out_features);
    }

    void validate() {
        dw::testutils::AssertTensorEqual(expected_weight, weight);
        dw::testutils::AssertTensorEqual(expected_bias, bias);
        dw::testutils::AssertTensorEqual(expected_weight_grad, weight_grad);
        dw::testutils::AssertTensorEqual(expected_bias_grad, bias_grad);
    }

    int batch_size, in_features, out_features;
    dw::Linear layer;

    dw::Tensor weight, bias, weight_grad, bias_grad;
    dw::Tensor expected_weight,
               expected_bias,
               expected_weight_grad,
               expected_bias_grad;

    dw::Tensor _output;
    dw::Tensor _grad_input;
};

struct SoftmaxHelper {
    SoftmaxHelper(int _batch_size, int _in_features)
        : batch_size(_batch_size),
          in_features(_in_features) {
    }

    dw::Placeholder operator()(dw::Placeholder in) {
        auto out = layer(in);
        _output.allocate(out.shape());
        _grad_input.allocate(in.shape());
        return out;
    }

    void forward(const dw::Tensor& input, dw::Tensor& output) {
        dw::reference::CPUSoftmaxForward(input.data(), output.data(), batch_size, in_features);
    }

    void backward(const dw::Tensor& output,
                  const dw::Tensor& grad_output,
                        dw::Tensor& grad_input) {
        dw::reference::CPUSoftmaxBackward(grad_output.data(), output.data(),
                                          grad_input.data(), batch_size, in_features);
    }

    int batch_size;
    int in_features;

    dw::Softmax layer;
    dw::Tensor _output;
    dw::Tensor _grad_input;
};

struct ResnetBlockHelper {
    ResnetBlockHelper(int out_channels, int _stride)
        : stride(_stride),
          downsample(stride > 1),
          conv0(out_channels, {3, 3}, {1, 1}, {stride, stride}),
          bn1(0.001, 0.05),
          conv3(out_channels, {3, 3}, {1, 1}, {1, 1}),
          bn4(0.001, 0.05),
          conv5(out_channels, {3, 3}, {1, 1}, {stride, stride}),
          bn6(0.001, 0.05) {
    }

    dw::Placeholder operator()(dw::Placeholder in) {
        auto out = conv0(in);

        out = bn1(out);
        out = relu2(out);
        out = conv3(out);
        out = bn4(out);

        // NB: Need to call bn6() and conv5() anyway in order to init parameters,
        // which participating in validate method.
        auto bn6_out = bn6(conv5(in));
        dw::Placeholder x = downsample ? bn6_out : in;

        x = add7(out, x);
        x = relu8(x);

        _output.allocate(x.shape());
        _grad_input.allocate(in.shape());

        return x;
    }

    void forward(const dw::Tensor& input, dw::Tensor& output) {
        conv0.forward(input, conv0._output);
        bn1.forward(conv0._output, bn1._output);
        relu2.forward(bn1._output, relu2._output);
        conv3.forward(relu2._output, conv3._output);
        bn4.forward(conv3._output, bn4._output);

        if (downsample) {
            conv5.forward(input, conv5._output);
            bn6.forward(conv5._output, bn6._output);
        }

        add7.forward(bn4._output, downsample ? bn6._output : input, add7._output);
        relu8.forward(add7._output, relu8._output);

        relu8._output.copyTo(output);
    }

    void backward(const dw::Tensor& input,
                  const dw::Tensor& output,
                  const dw::Tensor& grad_output,
                        dw::Tensor& grad_input) {
        relu8.backward(relu8._output, grad_output, relu8._grad_input);
        add7.backward(relu8._grad_input, add7._grad_input0, add7._grad_input1);

        if (downsample) {
            bn6.backward(add7._grad_input1, bn6._grad_input);
            conv5.backward(input, bn6._grad_input, conv5._grad_input);
        }

        bn4.backward(add7._grad_input0, bn4._grad_input);
        conv3.backward(relu2._output, bn4._grad_input, conv3._grad_input);
        relu2.backward(relu2._output, conv3._grad_input, relu2._grad_input);
        bn1.backward(relu2._grad_input, bn1._grad_input);
        conv0.backward(input, bn1._grad_input, conv0._grad_input);

        const auto* lhs = conv0._grad_input.data();
        const auto* rhs = downsample ? conv5._grad_input.data() : add7._grad_input1.data();
        for (int i = 0; i < conv0._grad_input.total(); ++i) {
            grad_input.data()[i] = lhs[i] + rhs[i];
        }
    }

    void validate() {
        conv0.validate();
        bn1.validate();
        conv3.validate();
        bn4.validate();
        conv5.validate();
        bn6.validate();
    }

    int stride;
    bool downsample;

    ConvolutionHelper conv0;
    BatchNorm2dHelper bn1;
    ReLUHelper        relu2;
    ConvolutionHelper conv3;
    BatchNorm2dHelper bn4;
    ConvolutionHelper conv5;
    BatchNorm2dHelper bn6;
    AddHelper         add7;
    ReLUHelper        relu8;

    dw::Tensor _output;
    dw::Tensor _grad_input;
};

struct ResnetBlockTest: public ::testing::Test {
    ResnetBlockTest()
        : block(in.shape()[1], 1),
          model(in, block(in)),
          output(model.outputs()[0].shape()),
          expected(output.shape()) {
        model.compile();
    }

    void forward_reference(const dw::Tensor& input, dw::Tensor& output) {
        block.forward(input, output);
    }

    void backward_reference(const dw::Tensor& input,
                            const dw::Tensor& output,
                            const dw::Tensor& grad_output) {
        block.backward(input, output, grad_output, block._grad_input);
    }

    void validate() {
        block.validate();
        dw::testutils::AssertTensorEqual(expected, output);
    }

    dw::Placeholder in{{2, 3, 4, 4}};
    ResnetBlockHelper block;
    dw::Model model;
    dw::Tensor output, expected;
};

// NB: Reduce resnet blocks because there
// is no such amount of memory on my laptop.
struct ResnetTest: public ::testing::Test {
    ResnetTest()
        : conv0(2, {3, 3}, {1, 1}, {1, 1}),
          bn1(0.001, 0.05),
          block3_0(2, 1),
          block3_1(2, 1),
          block4_0(4, 2),
          block4_1(4, 2),
          block5_0(8, 2),
          block5_1(8, 2),
          linear7(batch_size, 8, num_classes),
          softmax8(batch_size, num_classes) {
        auto out = conv0(in);
        out = bn1(out);
        out = relu2(out);
        out = block3_0(out);
        out = block3_1(out);
        out = block4_0(out);
        out = block4_1(out);
        out = block5_0(out);
        out = block5_1(out);
        out = pool6(out);
        out = linear7(out);
        out = softmax8(out);

        model = dw::Model(in, out);

        model.compile();

        output.allocate(model.outputs()[0].shape());
        expected.allocate(output.shape());
    }

    void forward_reference(const dw::Tensor& input, dw::Tensor& output) {
        conv0.forward(input, conv0._output);
        bn1.forward(conv0._output, bn1._output);
        relu2.forward(bn1._output, relu2._output);
        block3_0.forward(relu2._output, block3_0._output);
        block3_1.forward(block3_0._output, block3_1._output);
        block4_0.forward(block3_1._output, block4_0._output);
        block4_1.forward(block4_0._output, block4_1._output);
        block5_0.forward(block4_1._output, block5_0._output);
        block5_1.forward(block5_0._output, block5_1._output);
        pool6.forward(block5_1._output, pool6._output);
        linear7.forward(pool6._output, linear7._output);
        softmax8.forward(linear7._output, softmax8._output);

        softmax8._output.copyTo(output);
    }

    void backward_reference(const dw::Tensor& input,
                            const dw::Tensor& output,
                            const dw::Tensor& grad_output) {
        softmax8.backward(output, grad_output, softmax8._grad_input);
        linear7.backward(pool6._output, softmax8._grad_input, linear7._grad_input);
        pool6.backward(linear7._grad_input, pool6._grad_input);
        block5_1.backward(block5_0._output, block5_1._output, pool6._grad_input,    block5_1._grad_input);
        block5_0.backward(block4_1._output, block5_0._output, block5_1._grad_input, block5_0._grad_input);
        block4_1.backward(block4_0._output, block4_1._output, block5_0._grad_input, block4_1._grad_input);
        block4_0.backward(block3_1._output, block4_0._output, block4_1._grad_input, block4_0._grad_input);
        block3_1.backward(block3_0._output, block3_1._output, block4_0._grad_input, block3_1._grad_input);
        block3_0.backward(relu2._output, block3_0._output, block3_1._grad_input, block3_0._grad_input);
        relu2.backward(relu2._output, block3_0._grad_input, relu2._grad_input);
        bn1.backward(relu2._grad_input, bn1._grad_input);
        conv0.backward(input, bn1._grad_input, conv0._grad_input);
    }

    void validate() {
        conv0.validate();
        bn1.validate();
        block3_0.validate();
        block3_1.validate();
        block4_0.validate();
        block4_1.validate();
        block5_0.validate();
        block5_1.validate();
        linear7.validate();
        dw::testutils::AssertTensorEqual(expected, output);
    }

    int num_classes = 100;
    int batch_size  = 32;

    dw::Placeholder in{{batch_size, 3, 32, 32}};

    ConvolutionHelper   conv0;
    BatchNorm2dHelper   bn1;
    ReLUHelper          relu2;
    ResnetBlockHelper   block3_0;
    ResnetBlockHelper   block3_1;
    ResnetBlockHelper   block4_0;
    ResnetBlockHelper   block4_1;
    ResnetBlockHelper   block5_0;
    ResnetBlockHelper   block5_1;
    GlobalAVGPoolHelper pool6;
    LinearHelper        linear7;
    SoftmaxHelper       softmax8;

    dw::Model model;
    dw::Tensor output, expected;
};

TEST_F(ResnetBlockTest, Forward) {
    auto input = dw::Tensor::uniform(in.shape());

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    validate();
}

TEST_F(ResnetBlockTest, Backward) {
    auto input       = dw::Tensor::uniform(in.shape());
    auto grad_output = dw::Tensor::uniform(output.shape());

    // Deepworks
    model.forward(input, output);
    model.backward(input, output, grad_output);

    // Reference
    forward_reference(input, expected);
    backward_reference(input, expected, grad_output);

    // Assert
    validate();
}

TEST_F(ResnetTest, Forward) {
    auto input = dw::Tensor::uniform(in.shape());

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    validate();
}

TEST_F(ResnetTest, Backward) {
    auto input       = dw::Tensor::uniform(in.shape());
    auto grad_output = dw::Tensor::uniform(output.shape());

    // Deepworks
    model.forward(input, output);
    model.backward(input, output, grad_output);

    // Reference
    forward_reference(input, expected);
    backward_reference(input, expected, grad_output);

    // Assert
    validate();
}
