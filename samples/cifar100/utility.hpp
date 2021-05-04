#include <deepworks/deepworks.hpp>

#include <string>
#include <unordered_map>

namespace custom {

void normalize(deepworks::Tensor& t, float scale);

class CIFAR100Dataset : public deepworks::IDataset {
public:
    struct DataInfo {
        std::string path;
        int         label;
    };

    CIFAR100Dataset(const std::string& root);

    size_t size()                                                  override;
    deepworks::IDataset::OutShape shape()                          override;
    void pull(int idx, deepworks::Tensor& X, deepworks::Tensor& y) override;

private:
    std::vector<DataInfo> m_info;
    deepworks::Tensor m_image;

    static int label_counter_;
    static std::unordered_map<std::string, int> dir2label_;

    void HWC2CHW(deepworks::Tensor& image);
};

} // namespace custom
