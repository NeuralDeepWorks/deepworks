#include <iostream>

#include <deepworks/deepworks.hpp>

#include "utility.hpp"

namespace dw = deepworks;

static dw::Model buildCIFAR10Model(int batch_size) {
    int mid_features_first = 384;
    int mid_features_second = 192;
    int out_features = 10;

    std::array<int, 2> kernel_conv{5, 5};
    std::array<int, 2> padding_conv{2, 2};
    std::array<int, 2> stride_conv{1, 1};

    std::array<int, 2> kernel_pool{2, 2};
    std::array<int, 2> padding_pool{0, 0};
    std::array<int, 2> stride_pool{2, 2};

    dw::Placeholder in(dw::Shape{batch_size, 3, 32, 32});

    auto out = dw::Convolution(64, kernel_conv, padding_conv, stride_conv, "conv1")(in);
    out = dw::MaxPooling(kernel_pool, padding_pool, stride_pool, "pool2")(out);
    out = dw::ReLU("relu3")(out);

    out = dw::Linear(mid_features_first, "linear4")(out);
    out = dw::ReLU("relu5")(out);
    out = dw::BatchNorm1D(0.001, 0.05, "batchnorm1d6")(out);

    out = dw::Linear(mid_features_second, "linear7")(out);
    out = dw::ReLU("relu8")(out);
    out = dw::BatchNorm1D(0.001, 0.05, "batchnorm1d9")(out);

    out = dw::Linear(out_features, "linear10")(out);
    out = dw::Softmax("softmax11")(out);
    return {in, out};
}

int main(int argc, char *argv[]) {
    // Configuration
    std::string root = argv[1];
    auto train_dir   = root + "/train";
    auto test_dir    = root + "/test";
    int batch_size   = std::atoi(argv[2]);
    int num_epochs   = std::atoi(argv[3]);
    int freq         = std::atoi(argv[4]);

    // Define model
    auto model = buildCIFAR10Model(batch_size);
    model.compile();

    dw::optimizer::Adam opt(model.params(), 1e-3);
    dw::loss::CrossEntropyLoss criterion;

    deepworks::Tensor X, y;
    dw::DataLoader val_loader(std::make_shared<custom::CIFAR10Dataset>(test_dir) , batch_size, /*shuffle */ false);
    deepworks::Tensor predict(model.outputs()[0].shape());

    // NB: If path to pre-trained model is provided just run validation.
    if (argc == 6) {
        dw::load(model.state(), argv[5]);

        model.train(false);
        float acc    = 0.f;
        int val_iter = 0;

        // NB: Validation loop:
        while (val_loader.pull(X, y)) {
            model.forward(X, predict);
            acc += dw::metric::accuracy(predict, y);
            ++val_iter;
        }

        acc /= val_iter;
        std::cout << "Accuracy: " << acc << std::endl;
        return 0;
    }

    // NB: Otherwise train model.
    dw::DataLoader train_loader(std::make_shared<custom::CIFAR10Dataset>(train_dir), batch_size, /*shuffle */ true);
    deepworks::Tensor grad_output(model.outputs()[0].shape());
    for (int i = 0; i < num_epochs; ++i) {
        std::cout << "Epoch: " << i << std::endl;

        // NB: Reset train state
        model.train(true);
        float loss     = 0.f;
        int train_iter = 0;

        // NB: Training loop:
        while (train_loader.pull(X, y)) {
            model.forward(X, predict);

            loss += criterion.forward(predict, y);
            criterion.backward(predict, y, grad_output);
            model.backward(X, predict, grad_output);
            opt.step();

            ++train_iter;
            if (train_iter * batch_size % freq == 0) {
                std::cout << "Loss: " << loss / train_iter << std::endl;
            }
        }

        loss /= train_iter;
        std::cout << "Loss: " << loss << std::endl;

        // NB: Reset val state
        model.train(false);
        float acc    = 0.f;
        int val_iter = 0;

        // NB: Validation loop:
        while (val_loader.pull(X, y)) {
            model.forward(X, predict);
            acc += dw::metric::accuracy(predict, y);
            ++val_iter;
        }

        acc /= val_iter;
        std::cout << "Accuracy: " << acc << std::endl;
    }

    std::cout << "Model saved: cifar10_model.bin" << std::endl;
    dw::save(model.state(), "cifar10_model.bin");

    return 0;
}
