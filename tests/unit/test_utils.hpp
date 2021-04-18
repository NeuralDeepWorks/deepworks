#pragma once

#include <string>
#include <gtest/gtest.h>

namespace deepworks::testutils {
inline std::string GetTestDataPath() {
    return TEST_DATA_PATH;
}

inline float normL1(const deepworks::Tensor& actual, const deepworks::Tensor& expected) {
    auto* actual_p   = actual.data();
    auto* expected_p = expected.data();

    float norm_l1 = 0.f;
    for (size_t i = 0; i < actual.total(); i++) {
        norm_l1 += std::abs(actual_p[i] - expected_p[i]);
    }

    float norm_l1_relative = norm_l1 / expected.total();
    return norm_l1_relative;
}

inline void AssertEqual(float actual, float expected, float threshold = 1e-5) {
    ASSERT_NEAR(actual, expected, threshold);
}

inline void AssertEqual(const deepworks::Tensor& actual,
                        const deepworks::Tensor& expected,
                        float threshold = 1e-5) {
    auto* actual_p   = actual.data();
    auto* expected_p = expected.data();

    auto total = actual.total();
    for (int i = 0; i < total; ++i) {
        AssertEqual(actual_p[i], expected_p[i], threshold);
    }
}

inline void AssertTensorEqual(const deepworks::Tensor& actual,
                              const deepworks::Tensor& expected,
                              float threshold = 1e-5) {
    ASSERT_EQ(actual.shape()  , expected.shape());
    ASSERT_EQ(actual.strides(), expected.strides());

    float norm = normL1(actual, expected);
    EXPECT_LE(norm, 1e-5);
    AssertEqual(actual, expected, threshold);
}
} // deepworks::testutils
