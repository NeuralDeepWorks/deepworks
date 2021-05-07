#include "output_utils.hpp"

#include <vector>
#include <iomanip>
#include <cmath>

std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& vec) {
    if (vec.size() == 0) {
        return os;
    }
    os << '[';
    for (size_t index = 0; index + 1 < vec.size(); ++ index) {
        os << vec[index] << ", ";
    }
    os << vec[vec.size() - 1] << ']';
    return os;
}

void PrintEpochInformation(size_t current_epoch, size_t n_epochs, float train_loss, float val_accuracy) {
    std::cout << "- Epoch: " << std::right << std::setw(10) << current_epoch << "/" << n_epochs;
    std::cout << " | ";
    std::cout << "Train loss: " << std::setw(10) << std::left << train_loss;
    std::cout << " | ";
    std::cout << "Validation accuracy: " << std::setw(10) << std::left << val_accuracy << std::endl;
}

void PrintDelimiter(std::ostream& os) {
    for (int i = 0; i < OUTPUT_WIDTH; ++i) {
        os << "-";
    }
    os << std::endl;
}

void PrintHeader(const std::string& text, std::ostream& os) {
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

void PrintDatasetInfo(const std::string& name, size_t train_size, size_t validation_size, const std::vector<int64_t>& shape) {
    PrintHeader("Dataset info");
    std::cout << "- Name: " << name << std::endl;
    std::cout << "- Train size: " << train_size << std::endl;
    std::cout << "- Validation size: " << validation_size << std::endl;
    std::cout << "- Element shape: " << shape << std::endl;
}

void PrintBenchmarkResultsTable(const BenchmarkResults& deepworks_results, const BenchmarkResults& torch_results) {
    /* Table looks:
        =======================================
        ||     || Deepworks || Torch || Diff ||
        =======================================
    */
    const size_t width = OUTPUT_WIDTH + 4 * 2;
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
    auto get_procent_diff = [](auto target, auto to_compare, bool maximize) -> std::string {
        std::stringstream ss;
        if (target < to_compare && maximize) {
            ss << "-";
            std::swap(target, to_compare);
        } else if (target < to_compare && !maximize) {
            ss << "+";
            std::swap(target, to_compare);
        } else if (target > to_compare && !maximize) {
            ss << "-";
        } else {
            ss << "+";
        }

        double ratio = static_cast<double>(target) / to_compare;
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
    const std::string train_time_diff = get_procent_diff(deepworks_results.train_time, torch_results.train_time, false);
    print_row_info({"Train time", dw_train_time, torch_train_time, train_time_diff});

    const std::string dw_inference_time = std::to_string(deepworks_results.inference_time) + " ms";
    const std::string torch_inference_time = std::to_string(torch_results.inference_time) + " ms";
    const std::string inference_time_diff = get_procent_diff(deepworks_results.inference_time, torch_results.inference_time, false);
    print_row_info({"Inference time", dw_inference_time, torch_inference_time, inference_time_diff});

    const std::string dw_accuracy = std::to_string(deepworks_results.validation_accuracy);
    const std::string torch_accuracy = std::to_string(torch_results.validation_accuracy);
    const std::string accuracy_diff = get_procent_diff(deepworks_results.validation_accuracy, torch_results.validation_accuracy, true);
    print_row_info({"Accuracy", dw_accuracy, torch_accuracy, accuracy_diff});

    const std::string dw_loss = std::to_string(deepworks_results.train_loss);
    const std::string torch_loss = std::to_string(torch_results.train_loss);
    const std::string loss_diff = get_procent_diff(deepworks_results.train_loss, torch_results.train_loss, false);

    print_row_info({"Loss", dw_loss, torch_loss, loss_diff});
    print_row_delimiter();

    std::cout << std::endl;
}