#include <deepworks/deepworks.hpp>

namespace custom {

void normalize(deepworks::Tensor& t, float scale);

class Dataset : public deepworks::IDataset {
public:
    struct DataInfo {
        std::string path;
        int         label;
    };

    Dataset(const std::string& root);

    size_t size()                                                  override;
    deepworks::IDataset::OutShape shape()                          override;
    void pull(int idx, deepworks::Tensor& X, deepworks::Tensor& y) override;

private:
    std::vector<DataInfo> m_info;
};

} // namespace custom
