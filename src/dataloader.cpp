#include <deepworks/dataloader.hpp>
#include <deepworks/tensor.hpp>

#include "util/assert.hpp"
#include "util/generator.hpp"

#include <unordered_set>

static deepworks::Shape makeShape(int batch_size, const deepworks::Shape& shape) {
    deepworks::Shape new_shape(shape.size() + 1);
    new_shape[0] = batch_size;
    std::copy(shape.begin(), shape.end(), new_shape.begin() + 1);
    return new_shape;
}

static std::vector<int> randomIndices(int num, int upper_bound) {
    auto& gen = deepworks::detail::generator();
    std::unordered_set<int> unique;

    std::uniform_int_distribution<> dist(0, upper_bound);
    while (unique.size() != num) {
        unique.insert(dist(gen));
    }

    return {unique.begin(), unique.end()};
}

static std::vector<int> seqIndices(int num, int start) {
    std::vector<int> indices(num);
    std::iota(indices.begin(), indices.end(), start);
    return indices;
}

deepworks::DataLoader::DataLoader(deepworks::IDataset::Ptr dataset,
                                  int                      batch_size,
                                  bool                     shuffle)
    : m_dataset(dataset),
      m_batch_size(batch_size),
      m_shuffle(shuffle) {
}

bool deepworks::DataLoader::pull(deepworks::Tensor& X, deepworks::Tensor& y) {
    auto X_shape = makeShape(m_batch_size, m_dataset->shape().first);
    auto y_shape = makeShape(m_batch_size, m_dataset->shape().second);

    if (X.empty()) {
        X.allocate(X_shape);
    }

    if (y.empty()) {
        y.allocate(y_shape);
    }

    DeepWorks_Assert(X.shape() == X_shape);
    DeepWorks_Assert(y.shape() == y_shape);

    // Always skip if batch isn't full.
    if (m_pos + m_batch_size > m_dataset->size()) {
        reset();
        return false;
    }

    size_t X_batch_stride = X.strides()[0];
    size_t y_batch_stride = y.strides()[0];

    auto indices = m_shuffle ? randomIndices(m_batch_size, m_dataset->size() - 1)
                             : seqIndices(m_batch_size, m_pos);
    // NB: Fill batch
    for (int i = 0; i < m_batch_size; ++i) {
        deepworks::Tensor X_slice(m_dataset->shape().first , X.data() + i * X_batch_stride);
        deepworks::Tensor y_slice(m_dataset->shape().second, y.data() + i * y_batch_stride);
        m_dataset->pull(indices[i], X_slice, y_slice);
    }

    m_pos += m_batch_size;
    return true;
}

void deepworks::DataLoader::reset() {
    m_pos = 0;
}
