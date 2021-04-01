#include <deepworks/tensor.hpp>
#include <util/assert.hpp>

#include <numeric>
#include <algorithm>

namespace dw = deepworks;

struct dw::Tensor::Descriptor {
    dw::Strides            strides;
    dw::Shape              shape;
    std::shared_ptr<float> memory;
    dw::Tensor::Type*      data{nullptr};
    size_t                 total{0u};
};

dw::Strides calculateStrides(const dw::Shape& shape) {
    dw::Strides strides(shape.size());
    size_t initial_stride = 1;
    auto dim_it = shape.rbegin();
    for (auto stride_it = strides.rbegin(); stride_it != strides.rend(); ++stride_it, ++dim_it) {
        *stride_it = initial_stride;
        initial_stride *= *dim_it;
    }
    return strides;
}

dw::Tensor::Tensor() : m_descriptor(new Descriptor()) {
}

dw::Tensor::Tensor(const dw::Shape& shape) : Tensor() {
    allocate(shape);
}

dw::Tensor::Tensor(const Shape& shape, float* data) : Tensor() {
    init(shape);
    m_descriptor->data = data;
}

void dw::Tensor::init(const dw::Shape& shape) {
    bool have_negative_dim = std::any_of(shape.begin(), shape.end(),
                                         [](int dim) { return dim < 0; });
    DeepWorks_Assert(!have_negative_dim && "Cannot allocate tensor dynamic shape.");

    m_descriptor->shape   = shape;
    m_descriptor->strides = calculateStrides(shape);
    m_descriptor->total   = std::accumulate(shape.begin(),
                                            shape.end(),
                                            1,
                                            std::multiplies<>());
}

void dw::Tensor::copyTo(dw::Tensor& rhs) const {
    DeepWorks_Assert((m_descriptor->shape   == rhs.m_descriptor->shape    &&
                      m_descriptor->strides == rhs.m_descriptor->strides) &&
                      "Copy to another layout isn't supported");

    DeepWorks_Assert(!rhs.empty() && "copyTo: Output tensor should be allocated.");

    std::copy_n(m_descriptor->data, m_descriptor->total, rhs.m_descriptor->data);
}

void dw::Tensor::allocate(const dw::Shape& shape) {
    init(shape);

    m_descriptor->memory.reset(new Type[m_descriptor->total],
                               [](float* p){ delete[] p; });
    m_descriptor->data = m_descriptor->memory.get();
}

dw::Tensor::Type* dw::Tensor::data() const {
    return m_descriptor->data;
}

size_t dw::Tensor::total() const {
    return m_descriptor->total;
}

bool dw::Tensor::empty() const {
    return m_descriptor->total == 0;
}

const dw::Strides& dw::Tensor::strides() const {
    return m_descriptor->strides;
}

const dw::Shape& dw::Tensor::shape() const {
    return m_descriptor->shape;
}
