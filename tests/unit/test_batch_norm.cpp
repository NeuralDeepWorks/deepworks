#include <numeric>

#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

struct BatchNorm : public ::testing::Test {

    BatchNorm() : in(dw::Shape{batch_size, in_features}),
                  model(in, dw::BatchNorm1D(epsilon, alpha, "batchnorm1d")(in)) {
        model.compile();
    }

    void forward_reference(const dw::Tensor& input, dw::Tensor& output) {
        std::vector<float> ref_moving_mean(in_features);
        std::vector<float> ref_moving_var(in_features);

        const std::vector<float> gamma(in_features, 1);
        const std::vector<float> beta(in_features, 0);

        dw::reference::CPUBatchNorm1DForward(
                input.data(), output.data(),
                ref_moving_mean.data(), ref_moving_var.data(),
                true, epsilon, 0.5,
                gamma.data(), beta.data(), batch_size, in_features);
    }

    float epsilon = 0.001;
    float alpha = 0.05;
    int batch_size = 4;
    int in_features = 5;

    dw::Placeholder in;
    dw::Model model;

    dw::Tensor output{model.outputs()[0].shape()};
    dw::Tensor expected{output.shape()};
};

TEST_F(BatchNorm, Forward) {
    dw::Tensor input(in.shape());
    dw::initializer::uniform(input);

    // Deepworks
    model.forward(input, output);

    // Reference
    forward_reference(input, expected);

    // Assert
    dw::testutils::AssertTensorEqual(output, expected);
}
