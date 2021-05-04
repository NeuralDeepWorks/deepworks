#include "utility.hpp"

#include <algorithm>
#include <filesystem>

namespace dw = deepworks;
namespace fs = std::filesystem;

std::unordered_map<std::string, int> label2digit {
        {"airplane",   0},
        {"automobile", 1},
        {"bird",       2},
        {"cat",        3},
        {"deer",       4},
        {"dog",        5},
        {"frog",       6},
        {"horse",      7},
        {"ship",       8},
        {"truck",      9},
};

void custom::normalize(dw::Tensor& t, float scale) {
    std::transform(t.data(), t.data() + t.total(), t.data(),
            [&](float e) { return e / scale;});
}

custom::CIFAR10Dataset::CIFAR10Dataset(const std::string& root) {
    for (const auto & dir : fs::directory_iterator(root)) {
        int label = label2digit[dir.path().filename()];
        for (const auto& filename : fs::directory_iterator(dir.path())) {
            m_info.push_back(custom::CIFAR10Dataset::DataInfo{filename.path(), label});
        }
    }
}

size_t custom::CIFAR10Dataset::size() {
    return m_info.size();
}

dw::IDataset::OutShape custom::CIFAR10Dataset::shape() {
    return {dw::Shape{3, 32, 32}, dw::Shape{1}};
}

void custom::CIFAR10Dataset::pull(int idx, dw::Tensor& X, dw::Tensor& y) {
    dw::io::ReadImage(m_info[idx].path, X);
    HWC2CHW(X);
    normalize(X, 255);

    y.data()[0] = m_info[idx].label;
}

void custom::CIFAR10Dataset::HWC2CHW(deepworks::Tensor& image) {

    const auto& image_shape = image.shape();
    if (m_image.shape() != image_shape) {
        m_image = deepworks::Tensor(image_shape);
    }
    image.copyTo(m_image);

    auto source_data = m_image.data();
    auto target_data = image.data();

    size_t channel_stride = 32 * 32;

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
