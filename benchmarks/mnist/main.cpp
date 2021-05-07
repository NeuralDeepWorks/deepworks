#include <iostream>
#include <cmath>
#include "benchmark_dw.hpp"
#include "benchmark_torch.hpp"
#include "utils.hpp"

namespace dw = deepworks;


int main(int argc, char *argv[]) {
    // Configuration
    std::string root = argv[1];
    int batch_size   = std::atoi(argv[2]);
    int num_epochs   = std::atoi(argv[3]);

    std::string train_dir_path = root + "/train";
    std::string test_dir_path = root + "/test";
    
    auto dataset_train = TorchMNISTCustomDataset(train_dir_path)
                        .map(torch::data::transforms::Normalize<>(0, 255))
                        .map(torch::data::transforms::Stack<>());
                        
    auto dataset_val = TorchMNISTCustomDataset(test_dir_path)
                        .map(torch::data::transforms::Normalize<>(0, 255))
                        .map(torch::data::transforms::Stack<>());

    dw::DataLoader train_loader(std::make_shared<CustomDeepworksMnistDataset>(train_dir_path), batch_size, /*shuffle */ true);
    dw::DataLoader val_loader  (std::make_shared<CustomDeepworksMnistDataset>(test_dir_path) , batch_size, /*shuffle */ false);

    PrintDatasetInfo("MNIST", dataset_train.size().value(), dataset_val.size().value(), {28, 28});
    auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset_train), batch_size);
    auto data_loader_validation = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset_val), batch_size);

    size_t validation_size = dataset_val.size().value();

    PrintHeader("Torch");
    auto torch_result = executeTorchMNISTBenchmark(data_loader_train, data_loader_validation, num_epochs, validation_size);

    PrintHeader("Deepworks");
    auto dw_result = executeDeepworksMNISTBenchmark(train_loader, val_loader, num_epochs, batch_size);

    PrintBenchmarkResultsTable(dw_result, torch_result);
    return 0;
}
