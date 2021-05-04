#include <deepworks/deepworks.hpp>

namespace custom {

void normalize(deepworks::Tensor& t, float scale);

class CIFAR10Dataset : public deepworks::IDataset {
public:
    struct DataInfo {
        std::string path;
        int         label;
    };

    CIFAR10Dataset(const std::string& root);

    size_t size()                                                  override;
    deepworks::IDataset::OutShape shape()                          override;
    void pull(int idx, deepworks::Tensor& X, deepworks::Tensor& y) override;

private:
    std::vector<DataInfo> m_info;
    deepworks::Tensor m_image;

    void HWC2CHW(deepworks::Tensor& image);
};

} // namespace custom
