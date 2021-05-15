#include <iostream>

#include <deepworks/deepworks.hpp>

#include "utility.hpp"

namespace dw = deepworks;

static dw::Placeholder create_basic_block(dw::Placeholder x,
                                          int c_out,
                                          int stride = 1,
                                          bool downsample = false) {
    auto out = dw::Convolution(c_out, {3, 3}, {1, 1}, {stride, stride})(x);
    out = dw::BatchNorm2D()(out);
    out = dw::ReLU()(out);
    out = dw::Convolution(c_out, {3, 3}, {1, 1}, {1, 1})(out);
    out = dw::BatchNorm2D()(out);

    if (downsample) {
        x = dw::Convolution(c_out, {3, 3}, {1, 1}, {stride, stride})(x);
        x = dw::BatchNorm2D()(x);
    }

    x = dw::Add()(out, x);
    x = dw::ReLU()(x);
    return x;
}

static dw::Placeholder make_layer(dw::Placeholder x, int c_out, int stride=1) {
    // BasicBlock 1
    bool downsample = stride != 1 || x.shape()[1] != c_out;
    x = create_basic_block(x, c_out, stride, downsample);

    // BasicBlock 2
    x = create_basic_block(x, c_out);
    return x;
}

static dw::Model buildResnetModel(int batch_size) {
    int num_classes = 100;
    dw::Placeholder in({batch_size, 3, 32, 32});
    dw::Placeholder out;

    out = dw::Convolution(16, {3, 3}, {1, 1}, {1, 1})(in);
    out = dw::BatchNorm2D()(out);
    out = dw::ReLU()(out);

    out = make_layer(out, 16, 1);
    out = make_layer(out, 32, 2);
    out = make_layer(out, 64, 2);

    out = dw::GlobalAvgPooling()(out);
    out = dw::Linear(num_classes)(out);
    out = dw::Softmax()(out);

    return {in, out};
}

int main(int argc, char *argv[]) {
    // Configuration
    std::string mode = argv[1];
    std::string root = argv[2];
    auto train_dir   = root + "train";
    auto test_dir    = root + "test";
    int batch_size   = std::atoi(argv[3]);

    // NB: Used only for train mode.
    int num_epochs = -1;
    int freq       = -1;

    std::string model_path;
    if (mode == "train") {
        num_epochs   = std::atoi(argv[4]);
        freq         = std::atoi(argv[5]);
        model_path   = argv[6];
    } else if (mode == "test") {
        model_path   = argv[4];
    } else {
        throw std::logic_error("Unsupported mode: " + mode + "\n");
    }

    // Define model
    auto model = mode == "test" ? dw::load(model_path) : buildResnetModel(batch_size);
    model.compile();

    dw::optimizer::SGDMomentum opt(model.params(), 1e-4);
    dw::loss::CrossEntropyLoss criterion;

    deepworks::Tensor X, y;
    dw::DataLoader val_loader(std::make_shared<custom::CIFAR100Dataset>(test_dir) , batch_size, /*shuffle */ false);
    deepworks::Tensor predict(model.outputs()[0].shape());

    // NB: If path to pre-trained model is provided just run validation.
    if (mode == "test") {
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
    dw::DataLoader train_loader(std::make_shared<custom::CIFAR100Dataset>(train_dir), batch_size, /*shuffle */ true);
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

    std::cout << "Model saved: " << model_path << std::endl;
    dw::save(model, model_path);

    return 0;
}
