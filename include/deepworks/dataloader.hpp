#pragma once

#include <memory>

#include <deepworks/tensor.hpp>

namespace deepworks {

struct IDataset {
    using Ptr = std::shared_ptr<IDataset>;

    using OutShape = std::pair<Shape, Shape>;

    virtual size_t   size()                              = 0;
    virtual OutShape shape()                             = 0;
    virtual void     pull(int idx, Tensor& X, Tensor& y) = 0;
};

class DataLoader {
public:
    DataLoader(IDataset::Ptr dataset, int batch_size = 1, bool shuffle = true);
    bool pull(Tensor& X, Tensor& y);
    void reset();

private:
    IDataset::Ptr m_dataset;
    int           m_batch_size;
    bool          m_shuffle;
    int           m_pos = 0;
};

} // namespace dataloader
