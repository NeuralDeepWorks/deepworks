#pragma once

#include <vector>
#include <memory>
#include <iosfwd>

#include <deepworks/shape.hpp>

namespace deepworks {

using Strides = std::vector<size_t>;

class Tensor {
public:
    using Type = float;

    Tensor();
    explicit Tensor(const Shape &shape);

    void copyTo(Tensor tensor);

    Type *data();
    const Type *data() const;

    size_t total() const;
    void allocate(const Shape& shape);
    bool empty() const;
    const Strides &strides() const;
    const Shape &shape() const;

    ~Tensor() = default;

private:
    class Descriptor;
    std::shared_ptr<Descriptor> m_descriptor;
};

struct Tensor::Descriptor {
    Descriptor() = default;
    explicit Descriptor(const Shape& shape);

    void copyTo(Tensor::Descriptor& descriptor);
    void allocate(const Shape& shape);
    void calculateStrides(const Shape& shape);

    ~Descriptor();

    Strides m_strides;
    Shape m_shape;
    Type *m_data{nullptr};
    size_t m_total{0ul};
};

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

} // namespace deepworks
