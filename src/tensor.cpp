#include <tensor.hpp>
#include <numeric>
#include <algorithm>

namespace deepworks {

struct Tensor::Descriptor {
    Shape _shape;
    Strides _strides;
    Type *_data{nullptr};
};

Tensor::Tensor() {
    _descriptor = std::make_shared<Descriptor>();
}

Tensor::Tensor(const Shape &shape) {
    size_t n_elements = 1;
    for (auto dim : shape) {
        n_elements *= dim;
    }

    Strides strides(shape.size());

    size_t initial_stride = 1;
    auto dim_it = shape.rbegin();
    for (auto stride_it = strides.rbegin(); stride_it != strides.rend(); ++stride_it, ++dim_it) {
        *stride_it = initial_stride;
        initial_stride *= *dim_it;
    }
    _descriptor.reset(new Descriptor{shape, strides, new Type[n_elements]});
}

Tensor::Type *Tensor::data() {
    return _descriptor->_data;
}

size_t Tensor::totalElements() const {
    return std::accumulate(_descriptor->_shape.begin(),
                           _descriptor->_shape.end(),
                           1,
                           std::multiplies<>());
}

const Strides &Tensor::strides() const {
    return _descriptor->_strides;
}

const Shape &Tensor::shape() const {
    return _descriptor->_shape;
}

void Tensor::copyTo(Tensor &tensor) {
    if (&tensor == this) {
        return;
    }
    if (_descriptor->_data == nullptr) {
        throw std::runtime_error("Cannot copy not initialized tensor");

    }
    size_t n_elements = totalElements();

    if (tensor._descriptor->_data == nullptr) {
        tensor._descriptor.reset(new Descriptor{shape(),
                                                strides(),
                                                new Type[n_elements]});
    }

    if (n_elements != tensor.totalElements()) {
        throw std::runtime_error("Cannot copy tensor. Inconsistent number of elements");
    }
    tensor._descriptor->_shape = _descriptor->_shape;
    tensor._descriptor->_strides = _descriptor->_strides;

    std::copy_n(_descriptor->_data, n_elements, tensor._descriptor->_data);
}

Tensor::~Tensor() {
    if (_descriptor.use_count() == 1) {
        delete[] _descriptor->_data;
    }
}
} // namespace deepworks
