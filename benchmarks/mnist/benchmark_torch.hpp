#pragma once

#include <torch/torch.h>
#include <deepworks/deepworks.hpp>
#include <filesystem>
#include "utils.hpp"

namespace fs = std::filesystem;
namespace dw = deepworks;

struct TorchMNISTModel : torch::nn::Module {
    TorchMNISTModel() {
        linear1 = register_module("linear1", torch::nn::Linear(torch::nn::LinearOptions{kInFeatures, kMidFeatures}));
        relu = register_module("relu", torch::nn::ReLU());
        batch_norm = register_module("batch_norm1", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions{kMidFeatures}.eps(0.001).momentum(0.005)));
        linear2 = register_module("linear2", torch::nn::Linear(torch::nn::LinearOptions{kMidFeatures, kOutFeatures}));
    }
 
    torch::Tensor forward(torch::Tensor input) {
        auto x = torch::flatten(input, 1);
        x = linear1(x);
        x = relu(x);
        x = batch_norm(x);
        x = linear2(x);
        x = torch::log_softmax(x, 1);
        return x;
        
    }

    torch::nn::Linear linear1{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::BatchNorm1d batch_norm{nullptr};
    torch::nn::Linear linear2{nullptr};
};

class TorchMNISTCustomDataset : public torch::data::Dataset<TorchMNISTCustomDataset> {
public:
    TorchMNISTCustomDataset(const std::string& root) {
        for (const auto & dir : fs::directory_iterator(root)) {
            int label = std::stoi(dir.path().filename());
            for (const auto& filename : fs::directory_iterator(dir.path())) {
                m_info.push_back(DataInfo{filename.path(), label});
            }
        }
    }

    torch::data::Example<> get(size_t index) override {
        auto dw_image_tensor = dw::io::ReadImage(m_info[index].path);

        const auto X_options = torch::TensorOptions().dtype(torch::kFloat32);
        auto torch_X_tensor = torch::empty({kHeight, kWidth}, X_options);
        std::copy_n(dw_image_tensor.data(), dw_image_tensor.total(), torch_X_tensor.data_ptr<float>());

        const auto label_options = torch::TensorOptions().dtype(torch::kLong);
        auto torch_label_tensor = torch::empty({1}, label_options);
        torch_label_tensor.data_ptr<int64_t>()[0] = m_info[index].label;
        
        return {torch_X_tensor, torch_label_tensor};
    }

    torch::optional<size_t> size() const override{
        return m_info.size();
    }

    ~TorchMNISTCustomDataset() override = default;
private:
    std::vector<DataInfo> m_info;
};

template <class DataLoaderTrain, class DataLoaderTest>
BenchmarkResults executeTorchMNISTBenchmark(DataLoaderTrain& data_loader_train,
                                            DataLoaderTest& data_loader_validation,
                                            size_t epochs, size_t validation_size) {
    auto model = TorchMNISTModel();
    
    auto optimizer = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(1e-2));
    BenchmarkResults results;
    results.epochs = epochs;

    for (int i = 0; i < epochs; ++i) {
        model.train(true);
        float train_loss = 0.0f;
        int train_iter = 0;
        torch::Tensor loss;
        auto start_train = std::chrono::high_resolution_clock::now();
        for (const auto& batch : *data_loader_train) {
            optimizer.zero_grad();
            auto data = batch.data;
            auto targets = batch.target.reshape({data.size(0)});
            auto output = model.forward(data);

            loss = torch::nll_loss(output, targets);
            train_loss += loss.data_ptr<float>()[0];
            loss.backward();
            optimizer.step();

            ++train_iter;
        }
        auto end_train = std::chrono::high_resolution_clock::now();


        train_loss /= train_iter;
        model.train(false);
        int32_t correct = 0;
        auto start_val = std::chrono::high_resolution_clock::now();

        for (const auto& batch : *data_loader_validation) {
            auto data = batch.data;
            auto targets = batch.target.reshape({data.size(0)});
            auto output = model.forward(data);
            auto pred = output.argmax(1);
            correct += pred.eq(targets).sum().template item<int64_t>();
        }
        auto end_val = std::chrono::high_resolution_clock::now();
        float accuracy = static_cast<float>(correct) / validation_size;
        results.train_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_train - start_train).count();
        results.inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_val - start_val).count();
        PrintEpochInformation(i + 1, epochs, train_loss, accuracy);
        if (i + 1 == epochs) {
            results.train_loss = train_loss;
            results.validation_accuracy = accuracy;
        }
    }
    results.epochs = epochs;
    return results;
}
