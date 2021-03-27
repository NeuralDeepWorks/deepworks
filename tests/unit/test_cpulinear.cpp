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
    auto W = model.layers()[0].params()[0].data();
    auto b = model.layers()[0].params()[1].data();
    // Bias is zero by default, it isn't interesting for test.
    std::fill_n(b.data(), 10, 42);

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
    auto W = model.layers()[0].params()[0].data();
    auto b = model.layers()[0].params()[1].data();
    auto Wgrad = model.layers()[0].params()[0].grad();
    auto bgrad = model.layers()[0].params()[1].grad();
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
