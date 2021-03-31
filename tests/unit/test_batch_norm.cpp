#include <deepworks/deepworks.hpp>

#include "kernels_reference.hpp"
#include "../../src/runtime/cpu/kernels/kernels.hpp"
#include "test_utils.hpp"

namespace dw = deepworks;

TEST(LayerTests, CPUBatchNorm) {

    std::vector<float> vec(20);
    std::vector<float> vec2(20, 0);
    std::vector<float> ref_vec(20);
    std::vector<float> ref_vec2(20, 0);

    float count = 1.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            vec[i * 5 + j] = count;
            ref_vec[i * 5 + j] = count;
            count += 1.0;
        }
    }

    std::vector<float> vec3(5);
    std::vector<float> vec4(5);
    std::vector<float> ref_vec3(5);
    std::vector<float> ref_vec4(5);

    std::vector<float> vec5(20);
    std::vector<float> vec6(5);

    dw::ConstMatrix input(vec.data(), 4, 5);
    dw::Matrix output(vec2.data(), 4, 5);

    dw::Vector moving_mean(vec3.data(), 5);
    dw::Vector moving_var(vec4.data(), 5);

    dw::Matrix input_centered(vec5.data(), 4, 5);
    dw::Vector std(vec6.data(), 5);

    const std::vector<float> gamma(5, 0.2);
    const std::vector<float> beta(5, 0.1);
    dw::ConstVector Gamma(gamma.data(), 5);
    dw::ConstVector Beta(beta.data(), 5);

    dw::CPUBatchNormForward(input, output,
                            input_centered, std,
                            moving_mean, moving_var,
                            false, 0.5,
                            Gamma, Beta);


    dw::reference::CPUBatchNormForward(ref_vec.data(), ref_vec2.data(), ref_vec3.data(), ref_vec4.data(), false, 0.5,
                                       gamma.data(), beta.data(), 4, 5);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            EXPECT_FLOAT_EQ(vec2[i * 5 + j], ref_vec2[i * 5 + j]);
        }
    }
}

TEST(LayerTests, CPUBatchNormBackward) {

    std::vector<float> input_centered(20);

    float count = 1.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            input_centered[i * 5 + j] = count;
            count += 1.0;
        }
    }
    std::vector<float> std(5, 0.1);

    std::vector<float> grad_output(20, 3.0);
    std::vector<float> grad_input(20, 0.0);


    std::vector<float> gamma(5, 0.5);
    std::vector<float> gamma_grad(5, 0.0);
    std::vector<float> beta_grad(5, 0.0);

    dw::ConstMatrix Input_centered(input_centered.data(), 4, 5);
    dw::ConstVector Std(std.data(), 5);

    dw::ConstMatrix Grad_output(grad_output.data(), 4, 5);
    dw::Matrix Grad_input(grad_input.data(), 4, 5);

    dw::ConstVector Gamma(gamma.data(), 5);
    dw::Vector Gamma_grad(gamma_grad.data(), 5);
    dw::Vector Beta_grad(beta_grad.data(), 5);

    dw::CPUBatchNormBackward(Input_centered, Std,
                             Grad_output, Grad_input,
                             Gamma, Gamma_grad, Beta_grad);
}