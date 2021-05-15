#pragma once

#include <chrono>
#include <iomanip>
#include <unordered_map>

const std::unordered_map<std::string, int> label2digit = {
        {"airplane",   0},
        {"automobile", 1},
        {"bird",       2},
        {"cat",        3},
        {"deer",       4},
        {"dog",        5},
        {"frog",       6},
        {"horse",      7},
        {"ship",       8},
        {"truck",      9},
};

struct DataInfo {
    std::string path;
    int         label;
};

/* Dataset constants */
constexpr int kImageChannels = 3;
constexpr int kImageHeight = 32;
constexpr int kImageWidth = 32;

/* Network constants */
constexpr int kMidFeaturesFirst = 384;
constexpr int kMidFeaturesSecond = 192;
constexpr int kOutFeatures = 10;
constexpr int kFirstConvOutputChannels = 64;
constexpr int kSecondConvOutputChannels = 64;
constexpr float kBatchNormEps = 0.001;
constexpr float kBatchNormAlpha = 0.05;

constexpr std::array<int, 2> kKernelConv{5, 5};
constexpr std::array<int, 2> kPaddingConv{2, 2};
constexpr std::array<int, 2> kStrideConv{1, 1};

constexpr std::array<int, 2> kKernelPool{2, 2};
constexpr std::array<int, 2> kPaddingPool{0, 0};
constexpr std::array<int, 2> kStridePool{2, 2};
