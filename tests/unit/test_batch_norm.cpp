#include <numeric>

#include "kernels_reference.hpp"
#include "../../src/runtime/cpu/kernels/kernels.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPUBatchNorm) {
    const size_t rows = 4;
    const size_t cols = 5;
    const size_t size = rows * cols;

    std::vector<float> input(size);
    std::vector<float> ref_input(size);

    std::vector<float> output(size, 0);
    std::vector<float> ref_output(size, 0);

    std::iota(input.begin(), input.end(), 1);
    std::iota(ref_input.begin(), ref_input.end(), 1);

    std::vector<float> moving_mean(cols);
    std::vector<float> moving_var(cols);
    std::vector<float> ref_moving_mean(cols);
    std::vector<float> ref_moving_var(cols);

    std::vector<float> vec5(size);
    std::vector<float> vec6(cols);

    dw::ConstMatrix input_mat(input.data(), rows, cols);
    dw::Matrix output_mat(output.data(), rows, cols);

    dw::Vector moving_mean_vec(moving_mean.data(), cols);
    dw::Vector moving_var_vec(moving_var.data(), cols);

    dw::Matrix input_centered_mat(vec5.data(), rows, cols);
    dw::Vector std_vec(vec6.data(), cols);

    const std::vector<float> gamma(cols, 0.2);
    const std::vector<float> beta(cols, 0.1);
    dw::ConstVector gamma_vec(gamma.data(), cols);
    dw::ConstVector beta_vec(beta.data(), cols);

    const float epsilon = 0.01;

    dw::CPUBatchNorm1DForward(input_mat, output_mat,
                              input_centered_mat, std_vec,
                              moving_mean_vec, moving_var_vec,
                              true, epsilon, 0.5,
                              gamma_vec, beta_vec);

    dw::reference::CPUBatchNorm1DForward(ref_input.data(), ref_output.data(),
                                         ref_moving_mean.data(), ref_moving_var.data(),
                                         true, epsilon, 0.5,
                                         gamma.data(), beta.data(), rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ASSERT_NEAR(output[i * cols + j], ref_output[i * cols + j], 1e-7);
        }
    }
}

TEST(LayerTests, CPUBatchNormBackward) {
    const size_t rows = 4;
    const size_t cols = 5;
    const size_t size = rows * cols;

    std::vector<float> input_centered(size);

    std::iota(input_centered.begin(), input_centered.end(), 1);

    std::vector<float> std(cols, 0.1);

    std::vector<float> grad_output(size, 3.0);

    std::vector<float> grad_input(size, 0.0);
    std::vector<float> ref_grad_input(size, 0.0);

    const std::vector<float> gamma(cols, 0.5);

    std::vector<float> gamma_grad(cols, 0.0);
    std::vector<float> ref_gamma_grad(cols, 0.0);

    std::vector<float> beta_grad(cols, 0.0);
    std::vector<float> ref_beta_grad(cols, 0.0);

    dw::ConstMatrix input_centered_mat(input_centered.data(), rows, cols);
    dw::ConstVector std_vec(std.data(), cols);

    dw::ConstMatrix grad_output_mat(grad_output.data(), rows, cols);
    dw::Matrix grad_input_mat(grad_input.data(), rows, cols);

    dw::ConstVector gamma_vec(gamma.data(), cols);
    dw::Vector gamma_grad_vec(gamma_grad.data(), cols);
    dw::Vector beta_grad_vec(beta_grad.data(), cols);

    dw::CPUBatchNorm1DBackward(input_centered_mat, std_vec,
                               grad_output_mat, grad_input_mat,
                               gamma_vec, gamma_grad_vec, beta_grad_vec);

    dw::reference::CPUBatchNorm1DBackward(input_centered.data(), std.data(),
                                          grad_output.data(), ref_grad_input.data(),
                                          gamma.data(), ref_gamma_grad.data(), ref_beta_grad.data(),
                                          rows, cols);

    for (int j = 0; j < cols; j++) {
        ASSERT_NEAR(beta_grad[j], beta_grad[j], 1e-8);
        ASSERT_NEAR(gamma_grad[j], ref_gamma_grad[j], 1e-8);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ASSERT_NEAR(grad_input[i * cols + j], ref_grad_input[i * cols + j], 1e-8);
        }
    }
}
