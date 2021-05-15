#pragma once

#include "output_utils.hpp"
#include "utils.hpp"

#include <torch/torch.h>
#include <deepworks/deepworks.hpp>
#include <filesystem>

namespace fs = std::filesystem;
namespace dw = deepworks;

struct TorchCIFAR10Model : torch::nn::Module {
    TorchCIFAR10Model() {
        const auto conv1Options = torch::nn::Conv2dOptions{kImageChannels, kFirstConvOutputChannels, {kKernelConv[0], kKernelConv[1]}}
            .stride({kStrideConv[0], kStrideConv[1]})
            .padding({kPaddingConv[0], kPaddingConv[1]});
        const auto conv2Options = torch::nn::Conv2dOptions{kFirstConvOutputChannels, kSecondConvOutputChannels, {kKernelConv[0], kKernelConv[1]}}
            .stride({kStrideConv[0], kStrideConv[1]})
            .padding({kPaddingConv[0], kPaddingConv[1]});
        
        const auto poolOptions = torch::nn::MaxPool2dOptions({kKernelPool[0], kKernelPool[1]})
            .stride({kStridePool[0], kStridePool[1]})
            .padding({kPaddingPool[0], kPaddingPool[1]});

        conv1 = register_module("conv1", torch::nn::Conv2d(conv1Options));
        maxpool = register_module("maxpool", torch::nn::MaxPool2d(poolOptions));
        relu = register_module("relu", torch::nn::ReLU());
        conv2 = register_module("conv2", torch::nn::Conv2d(conv2Options));
        linear1 = register_module("linear1", torch::nn::Linear(torch::nn::LinearOptions{4096, kMidFeaturesFirst}));
        linear2 = register_module("linear2", torch::nn::Linear(torch::nn::LinearOptions{kMidFeaturesFirst, kMidFeaturesSecond}));
        linear3 = register_module("linear3", torch::nn::Linear(torch::nn::LinearOptions{kMidFeaturesSecond, kOutFeatures}));
        batchnorm1 = register_module("batchnorm1", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions{kMidFeaturesFirst}.eps(kBatchNormEps).momentum(kBatchNormAlpha)));
        batchnorm2 = register_module("batchnorm2", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions{kMidFeaturesSecond}.eps(kBatchNormEps).momentum(kBatchNormAlpha)));
    }
 
    torch::Tensor forward(torch::Tensor input) {

        auto x = conv1(input);
        x = maxpool(x);
        x = relu(x);

        x = conv2(x);
        x = maxpool(x);
        x = relu(x);

        x = torch::flatten(x, 1);
        x = linear1(x);
        x = relu(x);
        x = batchnorm1(x);

        x = linear2(x);
        x = relu(x);
        x = batchnorm2(x);

        x = linear3(x);
        x = torch::log_softmax(x, 1);

        return x;
    }
    torch::nn::ReLU relu{nullptr};
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::MaxPool2d maxpool{nullptr};
    torch::nn::Linear linear1{nullptr};
    torch::nn::Linear linear2{nullptr};
    torch::nn::Linear linear3{nullptr};
    torch::nn::BatchNorm1d batchnorm1{nullptr};
    torch::nn::BatchNorm1d batchnorm2{nullptr};
};

class TorchCIFAR10CustomDataset : public torch::data::Dataset<TorchCIFAR10CustomDataset> {
public:
    TorchCIFAR10CustomDataset(const std::string& root) {
        for (const auto & dir : fs::directory_iterator(root)) {
            int label = label2digit.find(dir.path().filename())->second;
            for (const auto& filename : fs::directory_iterator(dir.path())) {
                m_info.push_back(DataInfo{filename.path(), label});
            }
        }
    }

    torch::data::Example<> get(size_t index) override {
        auto dw_image_tensor = dw::io::ReadImage(m_info[index].path);

        const auto X_options = torch::TensorOptions().dtype(torch::kFloat32);
        auto torch_X_tensor = torch::empty({kImageChannels, kImageHeight, kImageWidth}, X_options);
        std::copy_n(dw_image_tensor.data(), dw_image_tensor.total(), torch_X_tensor.data_ptr<float>());

        const auto label_options = torch::TensorOptions().dtype(torch::kLong);
        auto torch_label_tensor = torch::empty({1}, label_options);
        torch_label_tensor.data_ptr<int64_t>()[0] = m_info[index].label;
        
        return {torch_X_tensor, torch_label_tensor};
    }

    torch::optional<size_t> size() const override{
        return m_info.size();
    }

    ~TorchCIFAR10CustomDataset() override = default;
private:
    std::vector<DataInfo> m_info;
    deepworks::Tensor m_image;
};

template <class DataLoaderTrain, class DataLoaderTest>
BenchmarkResults executeTorchCIFAR10Benchmark(DataLoaderTrain& data_loader_train,
                                            DataLoaderTest& data_loader_validation,
                                            size_t epochs, size_t validation_size) {
    auto model = TorchCIFAR10Model();
    
    auto optimizer = torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(1e-3));
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
