#include <gtest/gtest.h>

#include <numeric> // iota

#include <deepworks/tensor.hpp>
#include <deepworks/initializers.hpp>

#include "test_utils.hpp"

namespace dw = deepworks;

TEST(TensorTest, Allocate) {
    deepworks::Tensor src_tensor;
    EXPECT_TRUE(src_tensor.empty());
    EXPECT_EQ(src_tensor.shape(), (deepworks::Shape{}));
    ASSERT_EQ(src_tensor.data(), nullptr);

    src_tensor.allocate({1, 4, 16, 16});
    EXPECT_FALSE(src_tensor.empty());
    EXPECT_EQ(src_tensor.shape(), (deepworks::Shape{1, 4, 16, 16}));
    EXPECT_EQ(src_tensor.total(), 1 * 4 * 16 * 16);
    ASSERT_NE(src_tensor.data(), nullptr);
}

TEST(TensorTest, Strides) {
    struct CaseData {
        deepworks::Shape actual_shape;
        deepworks::Strides expected_strides;
    };
    auto test_cases = std::vector<CaseData>{
            {deepworks::Shape{1, 3, 224, 224}, deepworks::Strides{150528, 50176, 224, 1}},
            {deepworks::Shape{1, 200, 200, 300}, deepworks::Strides{12000000, 60000, 300, 1}},
            {deepworks::Shape{32, 200, 200, 300}, deepworks::Strides{12000000, 60000, 300, 1}},
            {deepworks::Shape{32, 3, 64, 64}, deepworks::Strides{12288, 4096, 64, 1}},
            {deepworks::Shape{4, 6, 12, 18}, deepworks::Strides{1296, 216, 18, 1}},
            {deepworks::Shape{4, 6, 12, 18}, deepworks::Strides{1296, 216, 18, 1}},
            {deepworks::Shape{2, 5}, deepworks::Strides{5, 1}},
            {deepworks::Shape{1}, deepworks::Strides{1}},
            {deepworks::Shape{}, deepworks::Strides{}},
    };
    for (const auto& tcase : test_cases) {
        auto tensor = deepworks::Tensor(tcase.actual_shape);
        EXPECT_EQ(tensor.strides(), tcase.expected_strides);
    }
}

TEST(TensorTest, Shape) {
    auto shapes = std::vector<deepworks::Shape>{
            deepworks::Shape{1, 3, 224, 224},
            deepworks::Shape{1, 200, 200, 300},
            deepworks::Shape{32, 200, 200, 300},
            deepworks::Shape{32, 3, 64, 64},
            deepworks::Shape{4, 6, 12, 18},
    };
    for (const auto& shape : shapes) {
        auto tensor = deepworks::Tensor(shape);
        EXPECT_EQ(tensor.shape(), shape);
    }
}

TEST(TensorTest, DefaultCtor) {
    deepworks::Tensor tensor;
    EXPECT_EQ(tensor.shape(), deepworks::Shape{});
    EXPECT_EQ(tensor.strides(), deepworks::Strides{});
    EXPECT_EQ(tensor.data(), nullptr);
    ASSERT_ANY_THROW(tensor.copyTo(tensor));
}

TEST(TensorTest, Reassignment) {
    deepworks::Tensor src_tensor({1, 3, 224, 224});
    EXPECT_FALSE(src_tensor.empty());
    deepworks::Tensor tensor;

    tensor = src_tensor;

    EXPECT_EQ(tensor.shape(), src_tensor.shape());
    EXPECT_EQ(tensor.strides(), src_tensor.strides());
    EXPECT_EQ(tensor.data(), src_tensor.data());
    EXPECT_FALSE(tensor.empty());
}

TEST(TensorTest, DynamicShape) {
    ASSERT_ANY_THROW(deepworks::Tensor src_tensor({-1, 3, 16, 16}));

    deepworks::Tensor tensor;
    ASSERT_ANY_THROW(tensor.allocate({1, 1, 1, -1, 1}));
}

TEST(TensorTest, CopyTo) {
    deepworks::Tensor src_tensor({1, 3, 224, 224});
    std::iota(src_tensor.data(), src_tensor.data() + src_tensor.total(), 0.f);

    deepworks::Tensor dst_tensor({1, 3, 224, 224});
    src_tensor.copyTo(dst_tensor);

    ASSERT_EQ(dst_tensor.shape(), src_tensor.shape());
    ASSERT_EQ(dst_tensor.strides(), src_tensor.strides());
    ASSERT_NE(dst_tensor.data(), src_tensor.data());
    ASSERT_FALSE(dst_tensor.empty());
}

TEST(TensorTest, CopyToNoThrow) {
    deepworks::Tensor src_tensor({1, 3, 224, 224});
    std::iota(src_tensor.data(), src_tensor.data() + src_tensor.total(), 0.f);
    deepworks::Tensor non_empty_tensor({1, 3, 16, 16});

    ASSERT_ANY_THROW(src_tensor.copyTo(non_empty_tensor));
}

TEST(TensorTest, CopyToEmptyThrow) {
    deepworks::Tensor src_tensor({1, 3, 224, 224});
    deepworks::Tensor dst_tensor;

    ASSERT_ANY_THROW(dst_tensor.copyTo(src_tensor));
}

TEST(TensorTest, FullView) {
    dw::Tensor t1(dw::Shape{32, 16});
    dw::initializer::uniform(t1);

    dw::Tensor t2(t1.shape(), t1.data());

    dw::testutils::AssertTensorEqual(t1, t2);
}

TEST(TensorTest, PartialView) {
    std::vector<float> raw = {1.f, 2.f, 3.f, 4.f};

    dw::Tensor t(dw::Shape{2}, raw.data());

    for (int i = 0; i < t.total(); ++i) {
        dw::testutils::AssertEqual(raw[i], t.data()[i]);
    }
}
