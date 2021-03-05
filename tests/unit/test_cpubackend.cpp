#include <gtest/gtest.h>

#include <deepworks/deepworks.hpp>

namespace dw = deepworks;

TEST(CPUBackend, ReLU) {
    int batch_size = 1;
    dw::Placeholder in(dw::Shape{1, 6});
    auto out = dw::ReLU("relu")(in);

    dw::Model model(in, out);
    model.compile(batch_size);

    std::vector<float> raw_in{-1, 2, 3, -5, 0, 10};
    dw::Tensor input(dw::Shape{1, static_cast<int>(raw_in.size())});
    std::copy(raw_in.begin(), raw_in.end(), input.data());

    auto output = model.forward({input});

    float* raw_out = output[0].data();

    std::cout << "OUTPUT: " << std::endl;
    for (int i = 0; i < raw_in.size(); ++i) {
        std::cout << raw_out[i] << " ";
    }
    std::cout << std::endl;
}
