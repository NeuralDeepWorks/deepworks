#include <iostream>
#include <cmath>
#include "benchmark_dw.hpp"
#include "benchmark_torch.hpp"

namespace dw = deepworks;

static constexpr int OUTPUT_WIDTH = 80;

void PrintDelimiter(std::ostream& os = std::cout) {
    for (int i = 0; i < OUTPUT_WIDTH; ++i) {
        os << "-";
    }
    os << std::endl;
}

void PrintHeader(const std::string& text, std::ostream& os = std::cout) {
    int width = OUTPUT_WIDTH + 4 * 2;
    for (int i = 0; i < width; ++i) {
        os << "*";
    }
    os << std::endl;
    std::cout << "*";
    int space_block_size = (width - 2 - text.size()) / 2;
    for (int i = 0; i < space_block_size; ++i) {
        os << " ";
    }
    std::cout << text;
        for (size_t i = 0; i < space_block_size; ++i) {
        os << " ";
    }
    os << "*";
    os << std::endl;
    for (int i = 0; i < width; ++i) {
        os << "*";
    }
    os << std::endl;
}

void PrintDatasetInfo(const std::string& name, size_t train_size, size_t validation_size, size_t image_height, size_t image_width) {
    PrintHeader("Dataset info");
    std::cout << "- Name: " << name << std::endl;
    std::cout << "- Train size: " << train_size << std::endl;
    std::cout << "- Validation size: " << validation_size << std::endl;
    std::cout << "- Image shape: " << image_height << "x" << image_width << std::endl;
}

void PrintBenchmarkResultsTable(const BenchmarkResults& deepworks_results, const BenchmarkResults& torch_results) {
    /* Table looks:
        =======================================
        ||     || Deepworks || Torch || Diff ||
        =======================================
    */
    const size_t width = 88;
    auto fill_spaces = [](size_t n) {
        for (size_t index = 0; index < n; ++index) {
            std::cout << " ";
        }
    };
    auto print_row_delimiter = [&]() {
        for (size_t index = 0; index < width; ++index) {
            std::cout << "=";
        }
        std::cout << std::endl;
    };
    auto print_row_info = [&fill_spaces](const std::array<std::string, 4>& text_cols) {
        size_t col_width = (width - 8) / text_cols.size();
        std::cout << "||";
        for (const auto& text : text_cols) {
            size_t space_length = (col_width - text.size()) / 2;
            fill_spaces(space_length);
            std::cout << text;

            if ((col_width - text.size()) % 2 == 1) {
                ++space_length;
            }

            fill_spaces(space_length);
            std::cout << "||";
        }
        std::cout << std::endl;
    };
    auto get_procent_diff = [](auto target, auto another) -> std::string {
        std::stringstream ss;
        if (target < another) {
            ss << "-";
            std::swap(target, another);
        } else {
            ss << "+";
        }

        double ratio = static_cast<double>(target) / another;
        double integer_part = 0;
        double fract_part = std::modf(ratio, &integer_part);
        integer_part -= 1;
        fract_part += integer_part;
        ss << std::fixed << std::setprecision(2) << fract_part * 100 << "%";
        return ss.str();
    };
    std::cout << std::endl;
    print_row_delimiter();
    print_row_info({"", "Deepworks", "Torch", "DW vs Torch"});
    print_row_delimiter();
    const std::string dw_epochs = std::to_string(deepworks_results.epochs);
    const std::string torch_epochs = std::to_string(torch_results.epochs);
    print_row_info({"Epochs", dw_epochs, torch_epochs, "None"});

    const std::string dw_train_time = std::to_string(deepworks_results.train_time) + " ms";
    const std::string torch_train_time = std::to_string(torch_results.train_time) + " ms";
    const std::string train_time_diff = get_procent_diff(deepworks_results.train_time, torch_results.train_time);
    print_row_info({"Train time", dw_train_time, torch_train_time, train_time_diff});

    const std::string dw_inference_time = std::to_string(deepworks_results.inference_time) + " ms";
    const std::string torch_inference_time = std::to_string(torch_results.inference_time) + " ms";
    const std::string inference_time_diff = get_procent_diff(deepworks_results.inference_time, torch_results.inference_time);
    print_row_info({"Inference time", dw_inference_time, torch_inference_time, inference_time_diff});

    const std::string dw_accuracy = std::to_string(deepworks_results.validation_accuracy);
    const std::string torch_accuracy = std::to_string(torch_results.validation_accuracy);
    const std::string accuracy_diff = get_procent_diff(deepworks_results.validation_accuracy, torch_results.validation_accuracy);
    print_row_info({"Accuracy", dw_accuracy, torch_accuracy, accuracy_diff});

    const std::string dw_loss = std::to_string(deepworks_results.train_loss);
    const std::string torch_loss = std::to_string(torch_results.train_loss);
    const std::string loss_diff = get_procent_diff(deepworks_results.train_loss, torch_results.train_loss);

    print_row_info({"Loss", dw_loss, torch_loss, loss_diff});
    print_row_delimiter();

    std::cout << std::endl;
}

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

    PrintDatasetInfo("MNIST", dataset_train.size().value(), dataset_val.size().value(), 28, 28);
    auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset_train), batch_size);
    auto data_loader_validation = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset_val), batch_size);

    size_t validation_size = dataset_val.size().value();

    PrintHeader("Torch");
    auto torch_result = executeTorchMNISTBenchmark(data_loader_train, data_loader_validation, num_epochs, validation_size);

    PrintHeader("Deepworks");
    auto dw_result = executeDeepworksMNISTBenchmark(train_loader, val_loader, num_epochs, batch_size);

    PrintBenchmarkResultsTable(torch_result, dw_result);
    return 0;
}
