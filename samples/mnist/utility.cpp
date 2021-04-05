#include "utility.hpp"

#include <algorithm>
#include <experimental/filesystem>

namespace dw = deepworks;
namespace fs = std::experimental::filesystem;

void custom::normalize(dw::Tensor& t, float scale) {
    std::transform(t.data(), t.data() + t.total(), t.data(),
            [&](float e) { return e / scale;});
}

custom::Dataset::Dataset(const std::string& root) {
    for (const auto & dir : fs::directory_iterator(root)) {
        int label = std::stoi(dir.path().filename());
        for (const auto& filename : fs::directory_iterator(dir.path())) {
            m_info.push_back(custom::Dataset::DataInfo{filename.path(), label});
        }
    }
}

size_t custom::Dataset::size() {
    return m_info.size();
}

dw::IDataset::OutShape custom::Dataset::shape() {
    return {dw::Shape{28 * 28}, dw::Shape{1}};
}

void custom::Dataset::pull(int idx, dw::Tensor& X, dw::Tensor& y) {
    dw::io::ReadImage(m_info[idx].path, X);
    normalize(X, 255);

    y.data()[0] = m_info[idx].label;
}
