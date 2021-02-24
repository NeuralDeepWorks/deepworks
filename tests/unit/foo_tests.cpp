#include <deepworks.hpp>
#include <gtest/gtest.h>

TEST(Sum, SimpleTest) {
    EXPECT_EQ(5, deepworks::sum(3, 2));
}
