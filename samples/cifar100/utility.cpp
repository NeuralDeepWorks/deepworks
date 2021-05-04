#include "utility.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
namespace dw = deepworks;
namespace fs = std::filesystem;

int custom::CIFAR100Dataset::label_counter_ = 0;
std::unordered_map<std::string, int> custom::CIFAR100Dataset::dir2label_ = {};

void custom::normalize(dw::Tensor& t, float scale) {
    std::transform(t.data(), t.data() + t.total(), t.data(),
                   [&](float e) { return e / scale;});
}

custom::CIFAR100Dataset::CIFAR100Dataset(const std::string& root) {
    for (const auto & dir : fs::directory_iterator(root)) {
        int label;
        auto res_find = dir2label_.find(dir.path().filename());
        if (res_find != dir2label_.end()) {
            label = res_find->second;
        } else {
            label = label_counter_;
            dir2label_[dir.path().filename()] = label_counter_;
            label_counter_++;
        }

        if (label == -1) {
            throw std::logic_error("Unsupported label\n");
        }

        for (const auto& filename : fs::directory_iterator(dir.path())) {
            m_info.push_back(custom::CIFAR100Dataset::DataInfo{filename.path(), label});
        }
    }
}

size_t custom::CIFAR100Dataset::size() {
    return m_info.size();
}

dw::IDataset::OutShape custom::CIFAR100Dataset::shape() {
    return {dw::Shape{3, 32, 32}, dw::Shape{1}};
}

void custom::CIFAR100Dataset::pull(int idx, dw::Tensor& X, dw::Tensor& y) {
    dw::io::ReadImage(m_info[idx].path, X);
    HWC2CHW(X);
    normalize(X, 255);

    y.data()[0] = m_info[idx].label;
}

void custom::CIFAR100Dataset::HWC2CHW(deepworks::Tensor& image) {

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
