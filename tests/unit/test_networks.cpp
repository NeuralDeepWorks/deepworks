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
        W0     = model.getLayer("linear0").params()[0].data();
        b0     = model.getLayer("linear0").params()[1].data();
        W1     = model.getLayer("linear2").params()[0].data();
        b1     = model.getLayer("linear2").params()[1].data();

        expected_params.emplace_back(dw::Tensor{W0.shape()});
        expected_params.emplace_back(dw::Tensor{b0.shape()});
        expected_params.emplace_back(dw::Tensor{W1.shape()});
        expected_params.emplace_back(dw::Tensor{b1.shape()});

        // NB: To easy access on specific parameter in tests.
        expected_W0 = expected_params[0].data();
        expected_b0 = expected_params[1].data();
        expected_W1 = expected_params[2].data();
        expected_b1 = expected_params[3].data();

        W0.copyTo(expected_W0);
        b0.copyTo(expected_b0);
        W1.copyTo(expected_W1);
        b1.copyTo(expected_b1);

        gradW0 = model.getLayer("linear0").params()[0].grad();
        gradb0 = model.getLayer("linear0").params()[1].grad();
        gradW1 = model.getLayer("linear2").params()[0].grad();
        gradb1 = model.getLayer("linear2").params()[1].grad();

        // NB: To easy access on specific parameter in tests.
        expected_gradW0 = expected_params[0].grad();
        expected_gradb0 = expected_params[1].grad();
        expected_gradW1 = expected_params[2].grad();
        expected_gradb1 = expected_params[3].grad();

        // NB: Not to compare trash against trash in tests
        dw::initializer::uniform(gradW0);
        dw::initializer::uniform(gradb0);
        dw::initializer::uniform(gradW1);
        dw::initializer::uniform(gradb1);

        gradW0.copyTo(expected_gradW0);
        gradb0.copyTo(expected_gradb0);
        gradW1.copyTo(expected_gradW1);
        gradb1.copyTo(expected_gradb1);

        grad_output.copyTo(expected_grad_output);

        loss = 0.0;
        expected_loss = loss;
    }

    // NB: in{batch_size, in_feautres} -> Linear0(mid_features) -> l0out{batch_size, mid_features}
    // -> ReLU1() -> r1out{batch_size, mid_features} -> Linear2(out_features) -> l2out{batch_size, out_features}
    // -> Softmax3() -> s3out{batch_size, out_features}

    void forward_reference(const dw::Tensor input, dw::Tensor& output) {
        dw::reference::CPULinearForward(input.data(), expected_W0.data(), linear_out0.data(),
                                        batch_size, in_features, mid_features);
        dw::reference::CPULinearAddBias(expected_b0.data(), linear_out0.data(), batch_size, mid_features);

        dw::reference::CPUReLUForward(linear_out0.data(), relu_out1.data(), relu_out1.total());

        dw::reference::CPULinearForward(relu_out1.data(), expected_W1.data(), linear_out2.data(),
                                        batch_size, mid_features, out_features);
        dw::reference::CPULinearAddBias(expected_b1.data(), linear_out2.data(), batch_size, out_features);

        dw::reference::CPUSoftmaxForward(linear_out2.data(), output.data(),
                                         linear_out2.shape()[0], linear_out2.shape()[1]);
    }

    void backward_reference(const dw::Tensor& input,
                            const dw::Tensor& output,
                            const dw::Tensor& grad_output) {
        dw::reference::CPUSoftmaxBackward(grad_output.data(), output.data(), linear2_gradout.data(),
                                          linear2_gradout.shape()[0], linear2_gradout.shape()[1]);

        dw::reference::CPULinearBackward(relu_out1.data(), expected_W1.data(), linear2_gradout.data(),
                                         expected_gradW1.data(), relu1_gradout.data(),
                                         batch_size, mid_features, out_features);

        dw::reference::CPULinearBiasBackward(linear2_gradout.data(), expected_gradb1.data(),
                                             batch_size, out_features);

        dw::reference::CPUReLUBackward(linear_out0.data(), relu1_gradout.data(), linear0_gradout.data(),
                                       batch_size, mid_features);

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
        dw::testutils::AssertTensorEqual(W0, expected_W0);
        dw::testutils::AssertTensorEqual(b0, expected_b0);
        // Validate gradients
        dw::testutils::AssertTensorEqual(gradW1, expected_gradW1);
        dw::testutils::AssertTensorEqual(gradb1, expected_gradb1);
        dw::testutils::AssertTensorEqual(gradW0, expected_gradW0);
        dw::testutils::AssertTensorEqual(gradb0, expected_gradb0);
        // Validate loss
        dw::testutils::AssertEqual(loss, expected_loss);
    }

    dw::Model buildModel() {
        dw::Placeholder in(dw::Shape{batch_size, in_features});
        auto out = dw::Linear(mid_features, "linear0")(in);
        out = dw::ReLU("relu1")(out);
        out = dw::Linear(out_features, "linear2")(out);
        out = dw::Softmax("softmax3")(out);
        return {in, out};
    }

    int in_features  = 28*28;
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

    dw::Tensor output{model.outputs()[0].shape()};
    dw::Tensor expected{output.shape()};
    dw::Tensor grad_output{output.shape()};
    dw::Tensor expected_grad_output{grad_output.shape()};

    dw::Tensor expected_W0, expected_b0, expected_W1, expected_b1;
    dw::Tensor expected_gradW0{dw::Shape{mid_features, in_features}},
               expected_gradb0{dw::Shape{mid_features}},
               expected_gradW1{dw::Shape{out_features, mid_features}},
               expected_gradb1{dw::Shape{out_features}};

    dw::Tensor W0, b0, W1, b1;
    dw::Tensor gradW0, gradb0, gradW1, gradb1;

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
