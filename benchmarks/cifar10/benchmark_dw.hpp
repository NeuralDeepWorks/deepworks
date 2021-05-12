#pragma once

#include "output_utils.hpp"
#include "utils.hpp"

#include <torch/torch.h>
#include <deepworks/deepworks.hpp>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;
namespace dw = deepworks;

static dw::Model buildCIFAR10Model(int batch_size) {


    dw::Placeholder in(dw::Shape{batch_size, kImageChannels, kImageHeight, kImageWidth});

    auto out = dw::Convolution(kFirstConvOutputChannels, kKernelConv, kPaddingConv, kStrideConv, "conv1")(in);
    out = dw::MaxPooling(kKernelPool, kPaddingPool, kStridePool, "pool2")(out);
    out = dw::ReLU("relu3")(out);

    out = dw::Convolution(kSecondConvOutputChannels, kKernelConv, kPaddingConv, kStrideConv, "conv4")(out);
    out = dw::MaxPooling(kKernelPool, kPaddingPool, kStridePool, "pool5")(out);
    out = dw::ReLU("relu6")(out);

    out = dw::Linear(kMidFeaturesFirst, "linear7")(out);
    out = dw::ReLU("relu8")(out);
    out = dw::BatchNorm1D(kBatchNormEps, kBatchNormAlpha, "batchnorm1d9")(out);

    out = dw::Linear(kMidFeaturesSecond, "linear10")(out);
    out = dw::ReLU("relu11")(out);
    out = dw::BatchNorm1D(kBatchNormEps, kBatchNormAlpha, "batchnorm1d12")(out);

    out = dw::Linear(kOutFeatures, "linear13")(out);
    out = dw::Softmax("softmax14")(out);
    return {in, out};
}

namespace {
void normalize(dw::Tensor& t, float scale) {
    std::transform(t.data(), t.data() + t.total(), t.data(),
            [&](float e) { return e / scale;});
}
}

class DeepworksCIFAR10Dataset : public deepworks::IDataset {
public:
    DeepworksCIFAR10Dataset(const std::string& root) {
        for (const auto & dir : fs::directory_iterator(root)) {
            int label = label2digit.find(dir.path().filename())->second;
            for (const auto& filename : fs::directory_iterator(dir.path())) {
                m_info.push_back(DataInfo{filename.path(), label});
            }
        }
    }

    size_t size() override {
        return m_info.size();
    }
    deepworks::IDataset::OutShape shape() override {
        return {dw::Shape{kImageChannels, kImageHeight, kImageWidth}, dw::Shape{1}};
    }
    void pull(int idx, deepworks::Tensor& X, deepworks::Tensor& y) override {
        dw::io::ReadImage(m_info[idx].path, X);
        HWC2CHW(X);
        normalize(X, 255);

        y.data()[0] = m_info[idx].label;
    }

private:
    std::vector<DataInfo> m_info;
    deepworks::Tensor m_image;

    void HWC2CHW(deepworks::Tensor& image) {

        const auto& image_shape = image.shape();
        if (m_image.shape() != image_shape) {
            m_image = deepworks::Tensor(image_shape);
        }
        image.copyTo(m_image);

        auto source_data = m_image.data();
        auto target_data = image.data();

        size_t channel_stride = kImageWidth * kImageHeight;

        size_t source_index = 0;
        size_t target_index = 0;
        while (source_index < image.total()) {
            // R channel
            target_data[target_index] = source_data[source_index];
            source_index++;

            // G channel
            target_data[target_index + channel_stride] = source_data[source_index];
            source_index++;

            // B channel
            target_data[target_index + 2 * channel_stride] = source_data[source_index];
            source_index++;

            target_index++;
        }
    }
};

BenchmarkResults executeDeepworksCIFAR10Benchmark(dw::DataLoader& train_loader,
                                                dw::DataLoader& val_loader,
                                                size_t epochs, size_t batch_size) {
    // Define model
    auto model = buildCIFAR10Model(batch_size);
    model.compile();

    dw::optimizer::Adam opt(model.params(), 1e-3);
    dw::loss::CrossEntropyLoss criterion;

    // Temprorary buffers
    deepworks::Tensor predict(model.outputs()[0].shape());
    deepworks::Tensor grad_output(model.outputs()[0].shape());
    deepworks::Tensor X_train, y_train;
    size_t train_miliseconds = 0;
    size_t validation_miliseconds = 0;
    BenchmarkResults results;
    results.epochs = epochs;
    for (int i = 0; i < epochs; ++i) {
        // NB: Reset train state
        model.train(true);
        float train_loss = 0.f;
        int train_iter = 0;

        // NB: Training loop:
        auto start_train = std::chrono::high_resolution_clock::now();

        while (train_loader.pull(X_train, y_train)) {
            model.forward(X_train, predict);

            train_loss += criterion.forward(predict, y_train);
            criterion.backward(predict, y_train, grad_output);
            model.backward(X_train, predict, grad_output);
            opt.step();

            ++train_iter;
        }
        auto end_train = std::chrono::high_resolution_clock::now();

        train_loss /= train_iter;

        // NB: Reset val state
        model.train(false);
        float acc    = 0.f;
        int val_iter = 0;

        // NB: Validation loop:
        auto start_val = std::chrono::high_resolution_clock::now();
        while (val_loader.pull(X_train, y_train)) {
            model.forward(X_train, predict);
            acc += dw::metric::accuracy(predict, y_train);
            ++val_iter;
        }
        auto end_val = std::chrono::high_resolution_clock::now();
        float accuracy = acc / val_iter;
        results.train_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_train - start_train).count();
        results.inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_val - start_val).count();
        PrintEpochInformation(i + 1, epochs, train_loss, accuracy);
        if (i + 1 == epochs) {
            results.train_loss = train_loss;
            results.validation_accuracy = accuracy;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    results.epochs = epochs;
    return results;
}
