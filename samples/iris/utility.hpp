#include <deepworks/deepworks.hpp>

#include <string>
#include <unordered_map>

namespace custom {

class IrisCSVDataset : public deepworks::IDataset {
public:
    struct IrisInfo {
        float sepal_length;
        float sepal_width;
        float petal_length;
        float petal_width;
        int   label;
    };

    IrisCSVDataset(const std::string& csv_path);

    size_t size()                                                  override;
    deepworks::IDataset::OutShape shape()                          override;
    void pull(int idx, deepworks::Tensor& X, deepworks::Tensor& y) override;

private:
    std::vector<IrisInfo> m_info;
    static std::unordered_map<std::string, int> m_iris2label;
};

} // namespace custom
