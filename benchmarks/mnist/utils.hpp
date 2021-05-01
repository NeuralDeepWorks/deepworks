#pragma once

#include <chrono>
#include <iomanip>

constexpr int kWidth = 28;
constexpr int kHeight = 28;
constexpr int kInFeatures = kWidth * kHeight;
constexpr int kMidFeatures = 100;
constexpr int kOutFeatures = 10;

void PrintEpochInformation(size_t current_epoch, size_t n_epochs, float train_loss, float val_accuracy) {
    std::cout << "- Epoch: " << std::right << std::setw(10) << current_epoch << "/" << n_epochs;
    std::cout << " | ";
    std::cout << "Train loss: " << std::setw(10) << std::left << train_loss;
    std::cout << " | ";
    std::cout << "Validation accuracy: " << std::setw(10) << std::left << val_accuracy << std::endl;
}

struct BenchmarkResults {
    size_t train_time = 0ul;
    size_t inference_time = 0ul;
    float train_loss = 0.0f;
    float validation_accuracy = 0.0f;
    size_t epochs = 0ul;
};

struct DataInfo {
    std::string path;
    int         label;
};
