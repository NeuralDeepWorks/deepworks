#include <iostream>

#include <deepworks/deepworks.hpp>

#include "utility.hpp"

namespace dw = deepworks;

static dw::Model buildMNISTModel(int batch_size) {
    int in_features  = 28*28;
    int mid_features = 100;
    int out_features = 10;

    dw::Placeholder in(dw::Shape{batch_size, in_features});
    auto out = dw::Linear(mid_features, "linear0")(in);
    out = dw::ReLU("relu1")(out);
    out = dw::BatchNorm1D(0.001, 0.05, "batchnorm1d")(out);
    out = dw::Linear(out_features, "linear2")(out);
    out = dw::Softmax("softmax3")(out);
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
    auto model = buildMNISTModel(batch_size);
    model.compile();

    dw::optimizer::SGDMomentum opt(model.params(), 1e-2);
    dw::loss::CrossEntropyLoss criterion;

    deepworks::Tensor X, y;
    dw::DataLoader val_loader(std::make_shared<custom::Dataset>(test_dir) , batch_size, /*shuffle */ false);
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
    dw::DataLoader train_loader(std::make_shared<custom::Dataset>(train_dir), batch_size, /*shuffle */ true);
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

    std::cout << "Model saved: mnist_model.bin" << std::endl;
    dw::save(model.state(), "mnist_model.bin");

    return 0;
}
