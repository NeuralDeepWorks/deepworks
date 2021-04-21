#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

#include <random>

namespace dw = deepworks;

namespace {

struct MNISTModel: public ::testing::Test {
    MNISTModel()
        : in(dw::Shape{batch_size, in_channels, in_features, in_features}),
          model(buildModel()) {
        model.compile();

        Wconv  = model.getLayer("conv0").params()[0].data();
        bconv  = model.getLayer("conv0").params()[1].data();
        W0     = model.getLayer("linear2").params()[0].data();
        b0     = model.getLayer("linear2").params()[1].data();
        gamma  = model.getLayer("batchnorm3").params()[0].data();
        beta   = model.getLayer("batchnorm3").params()[1].data();
        W1     = model.getLayer("linear4").params()[0].data();
        b1     = model.getLayer("linear4").params()[1].data();

        expected_params.emplace_back(dw::Tensor{Wconv});
        expected_params.emplace_back(dw::Tensor{bconv});
        expected_params.emplace_back(dw::Tensor{W0.shape()});
        expected_params.emplace_back(dw::Tensor{b0.shape()});
        expected_params.emplace_back(dw::Tensor{gamma.shape()});
        expected_params.emplace_back(dw::Tensor{beta.shape()});
        expected_params.emplace_back(dw::Tensor{W1.shape()});
        expected_params.emplace_back(dw::Tensor{b1.shape()});

        // NB: To easy access on specific parameter in tests.
        expected_Wconv  = expected_params[0].data();
        expected_bconv  = expected_params[1].data();
        expected_W0     = expected_params[2].data();
        expected_b0     = expected_params[3].data();
        expected_gamma  = expected_params[4].data();
        expected_beta   = expected_params[5].data();
        expected_W1     = expected_params[6].data();
        expected_b1     = expected_params[7].data();

        Wconv.copyTo(expected_Wconv);
        bconv.copyTo(expected_bconv);
        W0.copyTo(expected_W0);
        b0.copyTo(expected_b0);
        gamma.copyTo(expected_gamma);
        beta.copyTo(expected_beta);
        W1.copyTo(expected_W1);
        b1.copyTo(expected_b1);

        gradWconv = model.getLayer("conv0").params()[0].grad();
        gradbconv = model.getLayer("conv0").params()[1].grad();
        gradW0    = model.getLayer("linear2").params()[0].grad();
        gradb0    = model.getLayer("linear2").params()[1].grad();
        gradGamma = model.getLayer("batchnorm3").params()[0].grad();
        gradBeta  = model.getLayer("batchnorm3").params()[1].grad();
        gradW1    = model.getLayer("linear4").params()[0].grad();
        gradb1    = model.getLayer("linear4").params()[1].grad();

        // NB: To easy access on specific parameter in tests.
        expected_gradWconv = expected_params[0].grad();
        expected_gradbconv = expected_params[1].grad();
        expected_gradW0    = expected_params[2].grad();
        expected_gradb0    = expected_params[3].grad();
        expected_gradGamma = expected_params[4].grad();
        expected_gradBeta  = expected_params[5].grad();
        expected_gradW1    = expected_params[6].grad();
        expected_gradb1    = expected_params[7].grad();

        // NB: Not to compare trash against trash in tests
        dw::initializer::zeros(gradWconv);
        dw::initializer::zeros(gradbconv);
        dw::initializer::zeros(gradW0);
        dw::initializer::zeros(gradb0);
        dw::initializer::zeros(gradGamma);
        dw::initializer::zeros(gradBeta);
        dw::initializer::zeros(gradW1);
        dw::initializer::zeros(gradb1);

        gradWconv.copyTo(expected_gradWconv);
        gradbconv.copyTo(expected_gradbconv);
        gradW0.copyTo(expected_gradW0);
        gradb0.copyTo(expected_gradb0);
        gradGamma.copyTo(expected_gradGamma);
        gradBeta.copyTo(expected_gradBeta);
        gradW1.copyTo(expected_gradW1);
        gradb1.copyTo(expected_gradb1);

        // NB: Not to compare trash against trash in tests
        dw::initializer::zeros(grad_output);
        grad_output.copyTo(expected_grad_output);

        ref_input_centered = dw::Tensor{dw::Shape{batch_size, mid_features}};
        ref_std            = dw::Tensor{dw::Shape{mid_features}};
        ref_moving_mean    = dw::Tensor{dw::Shape{mid_features}};
        ref_moving_var     = dw::Tensor{dw::Shape{mid_features}};

        dw::initializer::zeros(ref_input_centered);
        dw::initializer::zeros(ref_std);
        dw::initializer::zeros(ref_moving_mean);
        dw::initializer::zeros(ref_moving_var);

        loss = 0.f;
        expected_loss = loss;
    }

    // NB: in{batch_size, in_feautres} -> Linear0(mid_features) -> l0out{batch_size, mid_features}
    // -> ReLU1() -> r1out{batch_size, mid_features}
    // -> BatchNorm2() -> b2out{batch_size, mid_features}
    // -> Linear3(out_features) -> l3out{batch_size, out_features}
    // -> Softmax4() -> s4out{batch_size, out_features}

    void forward_reference(const dw::Tensor input, dw::Tensor& output) {
        dw::reference::CPUConvolution2DForward(input, expected_Wconv, expected_bconv, conv_out0,
                                               kernel_conv, padding_conv, stride_conv);

//        dw::reference::CPUMaxPooling2DForward(conv_out0, maxpool_out1,
//                                              kernel_pool, padding_pool, stride_pool);

//        dw::reference::CPULinearForward(maxpool_out1.data(), expected_W0.data(), linear_out2.data(),
//                                        batch_size, linear_in_features, mid_features);
        dw::reference::CPULinearForward(conv_out0.data(), expected_W0.data(), linear_out2.data(),
                                        batch_size, linear_in_features, mid_features);
        dw::reference::CPULinearAddBias(expected_b0.data(), linear_out2.data(), batch_size, mid_features);

        dw::reference::CPUReLUForward(linear_out2.data(), relu_out3.data(), relu_out3.total());

        dw::reference::CPUBatchNorm1DForward(relu_out3, batch_norm_out4,
                                             ref_input_centered, ref_std,
                                             ref_moving_mean, ref_moving_var,
                                             train, epsilon, alpha,
                                             expected_gamma, expected_beta);

        dw::reference::CPULinearForward(batch_norm_out4.data(), expected_W1.data(), linear_out5.data(),
                                        batch_size, mid_features, out_features);
        dw::reference::CPULinearAddBias(expected_b1.data(), linear_out5.data(), batch_size, out_features);

        dw::reference::CPUSoftmaxForward(linear_out5.data(), output.data(),
                                         linear_out5.shape()[0], linear_out5.shape()[1]);
    }

    void backward_reference(const dw::Tensor& input,
                            const dw::Tensor& output,
                            const dw::Tensor& grad_output) {
        dw::reference::CPUSoftmaxBackward(grad_output.data(), output.data(), linear5_gradout.data(),
                                          linear5_gradout.shape()[0], linear5_gradout.shape()[1]);

        dw::reference::CPULinearBackward(batch_norm_out4.data(), expected_W1.data(), linear5_gradout.data(),
                                         expected_gradW1.data(), batch_norm4_gradout.data(),
                                         batch_size, mid_features, out_features);
        dw::reference::CPULinearBiasBackward(linear5_gradout.data(), expected_gradb1.data(),
                                             batch_size, out_features);

        dw::reference::CPUBatchNorm1DBackward(ref_input_centered, ref_std, batch_norm4_gradout,
                                              relu3_gradout, expected_gamma, expected_gradGamma, expected_gradBeta);

        dw::reference::CPUReLUBackward(linear_out2.data(), relu3_gradout.data(), linear2_gradout.data(),
                                       batch_size, mid_features);

        dw::reference::CPULinearBackward(conv_out0.data(), expected_W0.data(), linear2_gradout.data(),
                                         expected_gradW0.data(), conv0_gradout.data(),
                                         batch_size, linear_in_features, mid_features);
        dw::reference::CPULinearBiasBackward(linear2_gradout.data(), expected_gradb0.data(),
                                             batch_size, mid_features);

        dw::reference::CPUConvolution2DBackward(input, conv0_gradout,
                                                expected_Wconv, expected_bconv,
                                                expected_gradWconv,  expected_gradbconv,
                                                kernel_conv, padding_conv, stride_conv);
    }

    void validate(float threshold = 1e-5) {
        // Validate output
        dw::testutils::AssertTensorEqual(output, expected, threshold);
        // Validate grad outputs
        dw::testutils::AssertTensorEqual(grad_output, expected_grad_output, threshold);
        // Validate params
        dw::testutils::AssertTensorEqual(Wconv, expected_Wconv, threshold);
        dw::testutils::AssertTensorEqual(bconv, expected_bconv, threshold);
        dw::testutils::AssertTensorEqual(W1, expected_W1, threshold);
        dw::testutils::AssertTensorEqual(b1, expected_b1, threshold);
        dw::testutils::AssertTensorEqual(gamma, expected_gamma, threshold);
        dw::testutils::AssertTensorEqual(beta, expected_beta, threshold);
        dw::testutils::AssertTensorEqual(W0, expected_W0, threshold);
        dw::testutils::AssertTensorEqual(b0, expected_b0, threshold);
        // Validate gradients
        dw::testutils::AssertTensorEqual(gradWconv, expected_gradWconv, threshold);
        dw::testutils::AssertTensorEqual(gradbconv, expected_gradbconv, threshold);
        dw::testutils::AssertTensorEqual(gradW1, expected_gradW1, threshold);
        dw::testutils::AssertTensorEqual(gradb1, expected_gradb1, threshold);
        dw::testutils::AssertTensorEqual(gradGamma, expected_gradGamma, threshold);
        dw::testutils::AssertTensorEqual(gradBeta, expected_gradBeta, threshold);
        dw::testutils::AssertTensorEqual(gradW0, expected_gradW0, threshold);
        dw::testutils::AssertTensorEqual(gradb0, expected_gradb0, threshold);
        // Validate loss
        dw::testutils::AssertEqual(loss, expected_loss, threshold);
    }

    dw::Model buildModel() {
        dw::Placeholder in(dw::Shape{batch_size, in_channels, in_features, in_features});
        auto out = dw::Convolution(out_channels, kernel_conv, padding_conv, stride_conv, "conv0")(in);
//        out = dw::MaxPooling(kernel_pool, padding_pool, stride_pool, "pool1")(out);
        out = dw::Linear(mid_features, "linear2")(out);
        out = dw::ReLU("relu3")(out);
        out = dw::BatchNorm1D(epsilon, alpha, "batchnorm3")(out);
        out = dw::Linear(out_features, "linear4")(out);
        out = dw::Softmax("softmax5")(out);
        return {in, out};
    }

    bool train = true;

    std::array<int, 2> kernel_conv{5, 5};
    std::array<int, 2> padding_conv{2, 2};
    std::array<int, 2> stride_conv{1, 1};
    std::array<int, 2> kernel_pool{2, 2};
    std::array<int, 2> padding_pool{0, 0};
    std::array<int, 2> stride_pool{2, 2};
    int in_channels  = 1;
    int out_channels = 1;
    int in_features  = 28;
    int pool_features = 28;
    int mid_features = 100;
    int out_features = 10;
    int batch_size   = 8;
    int linear_in_features = out_channels * pool_features * pool_features;

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
    dw::Tensor conv_out0      {dw::Shape{batch_size, out_channels, in_features, in_features}};
    dw::Tensor maxpool_out1   {dw::Shape{batch_size, out_channels, pool_features, pool_features}};
    dw::Tensor linear_out2    {dw::Shape{batch_size, mid_features}};
    dw::Tensor relu_out3      {dw::Shape{batch_size, mid_features}};
    dw::Tensor batch_norm_out4{dw::Shape{batch_size, mid_features}};
    dw::Tensor linear_out5    {dw::Shape{batch_size, out_features}};
    // NB: Intermediate tensors (Backward)
    dw::Tensor linear5_gradout    {dw::Shape{batch_size, out_features}};
    dw::Tensor batch_norm4_gradout{dw::Shape{batch_size, mid_features}};
    dw::Tensor relu3_gradout      {dw::Shape{batch_size, mid_features}};
    dw::Tensor linear2_gradout    {dw::Shape{batch_size, mid_features}};
    dw::Tensor maxpool1_gradout   {dw::Shape{batch_size, out_channels, pool_features, pool_features}};
    dw::Tensor conv0_gradout      {dw::Shape{batch_size, out_channels, in_features, in_features}};
    dw::Tensor grad_input         {dw::Shape{batch_size, out_channels, in_features, in_features}};

    dw::Tensor output{model.outputs()[0].shape()};
    dw::Tensor expected{output.shape()};
    dw::Tensor grad_output{output.shape()};
    dw::Tensor expected_grad_output{grad_output.shape()};

    dw::Tensor expected_Wconv, expected_bconv,
               expected_W0, expected_b0,
               expected_gamma, expected_beta,
               expected_W1, expected_b1;
    dw::Tensor expected_gradWconv, expected_gradbconv,
               expected_gradW0, expected_gradb0,
               expected_gradGamma, expected_gradBeta,
               expected_gradW1,  expected_gradb1;

    dw::Tensor Wconv, bconv, W0, b0, gamma, beta, W1, b1;
    dw::Tensor gradWconv, gradbconv, gradW0, gradb0, gradGamma, gradBeta, gradW1, gradb1;

    dw::Parameters expected_params;

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
    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    validate();
}

TEST_F(MNISTModel, ForwardTrainFalse) {
    // Init
    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    train = false;
    model.train(train);

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    validate();
}

TEST_F(MNISTModel, Backward) {
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
    validate();
}

TEST_F(MNISTModel, TrainLoopSmoke) {
    size_t num_imgs = 100u;
    int num_epochs  = 10;
    auto loader     = dw::DataLoader(std::make_shared<Dataset>(num_imgs), batch_size, false);

    dw::Tensor X, y;
    dw::Tensor X4d(in.shape());

    // Deepworks train loop
    dw::loss::CrossEntropyLoss criterion;
    dw::optimizer::SGD opt(model.params(), 0.01);
    for (int i = 0; i < num_epochs; ++i) {
        while (loader.pull(X, y)) {
            std::copy_n(X.data(), X.total(), X4d.data());
            model.forward(X4d, output);
            loss = criterion.forward(output, y);
            criterion.backward(output, y, grad_output);
            model.backward(X4d, output, grad_output);
            opt.step();
        }
    }
    
    loader.reset();

    // Reference train loop
    for (int i = 0; i < num_epochs; ++i) {
        while (loader.pull(X, y)) {
            std::copy_n(X.data(), X.total(), X4d.data());
            forward_reference(X4d, expected);
            expected_loss = dw::reference::CPUCrossEntropyLossForward(expected, y);
            dw::reference::CPUCrossEntropyLossBackward(expected, y, expected_grad_output);
            backward_reference(X4d, expected, expected_grad_output);
            dw::reference::SGDStep(expected_params, opt.get_lr());
        }
    }

    // Assert
    validate(1e-4);
}
