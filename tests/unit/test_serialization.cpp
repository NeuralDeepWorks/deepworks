#include <gtest/gtest.h>

#include <deepworks/deepworks.hpp>

#include "test_utils.hpp"

namespace dw = deepworks;

struct SimpleModelSerializationTest: public ::testing::Test {
    dw::Model buildModel() {
        dw::Placeholder in(dw::Shape{batch_size, in_features});
        auto out = dw::Linear(out_features, "linear0")(in);
        return {in, out};
    }

    int batch_size   = 1;
    int in_features  = 10;
    int out_features = 2;
};

TEST_F(SimpleModelSerializationTest, SaveStateDict) {
    auto m1 = buildModel();
    dw::save(m1.state(), "state.bin");

    auto m2 = buildModel();
    dw::load(m2.state(), "state.bin");

    for (const auto& [name, tensor] : m1.state()) {
        dw::testutils::AssertTensorEqual(tensor, m2.state().at(name));
    }
}
