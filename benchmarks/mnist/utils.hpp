#pragma once

#include <chrono>
#include <iomanip>

constexpr int kWidth = 28;
constexpr int kHeight = 28;
constexpr int kInFeatures = kWidth * kHeight;
constexpr int kMidFeatures = 100;
constexpr int kOutFeatures = 10;

struct DataInfo {
    std::string path;
    int         label;
};
