#include "utils.hpp"
#include <torch/torch.h>

#include <deepworks/deepworks.hpp>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;
namespace dw = deepworks;

deepworks::Model buildMNISTModel(int batch_size) {
    dw::Placeholder in(dw::Shape{batch_size, kInFeatures});
    auto out = dw::Linear(kMidFeatures, "linear0")(in);
    out = dw::ReLU("relu1")(out);
    out = dw::BatchNorm1D(0.001, 0.05, "batchnorm1d")(out);
    out = dw::Linear(kOutFeatures, "linear2")(out);
    out = dw::Softmax("softmax3")(out);
    return {in, out};
}

namespace {
void normalize(dw::Tensor& t, float scale) {
    std::transform(t.data(), t.data() + t.total(), t.data(),
            [&](float e) { return e / scale;});
}
}

class CustomDeepworksMnistDataset : public deepworks::IDataset {
public:
    CustomDeepworksMnistDataset(const std::string& root) {
        for (const auto & dir : fs::directory_iterator(root)) {
            int label = std::stoi(dir.path().filename());
            for (const auto& filename : fs::directory_iterator(dir.path())) {
                m_info.push_back(DataInfo{filename.path(), label});
            }
        }
    }

    size_t size() override {
        return m_info.size();
    }
    deepworks::IDataset::OutShape shape() override {
        return {dw::Shape{28 * 28}, dw::Shape{1}};
    }
    void pull(int idx, deepworks::Tensor& X, deepworks::Tensor& y) override {
        dw::io::ReadImage(m_info[idx].path, X);
        normalize(X, 255);
        y.data()[0] = m_info[idx].label;
    }

private:
    std::vector<DataInfo> m_info;
};

BenchmarkResults executeDeepworksMNISTBenchmark(dw::DataLoader& train_loader,
                                                dw::DataLoader& val_loader,
                                                size_t epochs, size_t batch_size) {
    // Define model
    auto model = buildMNISTModel(batch_size);
    model.compile();

    dw::optimizer::SGD opt(model.params(), 1e-2);
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
