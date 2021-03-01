#include <tensor.hpp>
#include <numeric>
#include <algorithm>

namespace deepworks {

Tensor::Descriptor::Descriptor(const Shape& shape) : m_shape(shape) {
    m_strides.resize(shape.size());

    size_t initial_stride = 1;
    auto dim_it = shape.rbegin();
    for (auto stride_it = m_strides.rbegin(); stride_it != m_strides.rend(); ++stride_it, ++dim_it) {
        *stride_it = initial_stride;
        initial_stride *= *dim_it;
    }
}

void Tensor::Descriptor::copyTo(Tensor::Descriptor& descriptor) {
    if (!descriptor.m_created) {
        throw std::runtime_error("copyTo: Output tensor must be allocated");
    }

    if (this == &descriptor) {
        throw std::runtime_error("Tensor cannot copy itself.");
    }

    size_t n_elements = total();
    size_t dst_n_elements = descriptor.total();

    if (n_elements != dst_n_elements) {
        throw std::runtime_error("Cannot copy tensor: inconsistent number of elements");
    }

    m_shape = descriptor.m_shape;
    m_strides = descriptor.m_strides;

    std::copy_n(m_data, n_elements, descriptor.m_data);
}

size_t Tensor::Descriptor::total() {
    return std::accumulate(m_shape.begin(),
                           m_shape.end(),
                           1,
                           std::multiplies<>());
}

void Tensor::Descriptor::create() {
    int total_elements = total();
    if (total_elements == 0) {
        throw std::runtime_error("Tensor is empty. Cannot allocate.");
    }
    if (m_created) {
        throw std::runtime_error("Tensor already allocated, cannot call create() twice.");
    }
    m_data = new Type[total_elements];
    m_created = true;
}

Tensor::Descriptor::~Descriptor() {
    delete [] m_data;
}

/* Tensor */
Tensor::Tensor() {
    m_descriptor.reset(new Descriptor());
}

Tensor::Tensor(const Shape &shape) {
    m_descriptor.reset(new Descriptor(shape));
}

void Tensor::copyTo(Tensor &tensor) {
    m_descriptor->copyTo(*(tensor.m_descriptor));
}

Tensor::Type *Tensor::data() {
    return m_descriptor->m_data;
}

size_t Tensor::total() const {
    return m_descriptor->total();
}

void Tensor::create() {
    m_descriptor->create();
}

bool Tensor::isCreated() const {
    return m_descriptor->m_created;
}

const Strides &Tensor::strides() const {
    return m_descriptor->m_strides;
}

const Shape &Tensor::shape() const {
    return m_descriptor->m_shape;
}

} // namespace deepworks
