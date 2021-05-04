#include "utility.hpp"

namespace dw = deepworks;

namespace {

std::vector<std::string> SplitLineByComma(const std::string& line) {
    std::vector<std::string> result;

    size_t find_pos;
    size_t start_pos = 0;
    while ((find_pos = line.find(',', start_pos)) != std::string::npos) {
        result.push_back(line.substr(start_pos, find_pos - start_pos));
        start_pos = find_pos + 1;
    }
    // push back line part after last comma
    result.push_back(line.substr(start_pos));

    return std::move(result);
}

} // namespace

std::unordered_map<std::string, int> custom::IrisCSVDataset::m_iris2label = {
        {"setosa",     0},
        {"versicolor", 1},
        {"virginica",  2}
};

custom::IrisCSVDataset::IrisCSVDataset(const std::string& csv_path) {

    std::ifstream iris_dataset;
    iris_dataset.open(csv_path);

    std::string iris;
    // read headers
    std::getline(iris_dataset, iris);

    while (std::getline(iris_dataset, iris)) {
        auto features = SplitLineByComma(iris);

        float sepal_length = std::atof(features[0].c_str());
        float sepal_width  = std::atof(features[1].c_str());
        float petal_length = std::atof(features[2].c_str());
        float petal_width  = std::atof(features[3].c_str());

        int label = m_iris2label[features[4]];
       
        m_info.push_back(IrisInfo{sepal_length, sepal_width, petal_length, petal_width, label});
    }
}

size_t custom::IrisCSVDataset::size() {
    return m_info.size();
}

dw::IDataset::OutShape custom::IrisCSVDataset::shape() {
    return {dw::Shape{4}, dw::Shape{1}};
}

void custom::IrisCSVDataset::pull(int idx, dw::Tensor& X, dw::Tensor& y) {
    X.data()[0] = m_info[idx].sepal_length;
    X.data()[1] = m_info[idx].sepal_width;
    X.data()[2] = m_info[idx].petal_length;
    X.data()[3] = m_info[idx].petal_width;

    y.data()[0] = m_info[idx].label;
}
