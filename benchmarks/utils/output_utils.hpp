#pragma once

#include <iostream>
#include <vector>

static constexpr int OUTPUT_WIDTH = 80;

struct BenchmarkResults {
    size_t epochs = 0ul;
    size_t train_time = 0ul;
    size_t inference_time = 0ul;
    float train_loss = 0.0f;
    float validation_accuracy = 0.0f;
};

void PrintEpochInformation(size_t current_epoch, size_t n_epochs, float train_loss, float val_accuracy);

void PrintDelimiter(std::ostream& os = std::cout);

void PrintHeader(const std::string& text, std::ostream& os = std::cout);

void PrintDatasetInfo(const std::string& name, size_t train_size, size_t validation_size, const std::vector<int64_t>& shape);

void PrintBenchmarkResultsTable(const BenchmarkResults& deepworks_results, const BenchmarkResults& torch_results);