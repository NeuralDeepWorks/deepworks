#pragma once

#include <vector>
#include <memory>

#include <deepworks/shape.hpp>

namespace deepworks {

using Strides = std::vector<size_t>;

class Tensor {
public:
    using Type = float;

    Tensor();
    Tensor(const Shape& shape, float* data);
    explicit Tensor(const Shape& shape);

    Type*           data()    const;
    size_t          total()   const;
    bool            empty()   const;
    const  Strides& strides() const;
    const  Shape&   shape()   const;

    void copyTo(Tensor& tensor) const;
    void allocate(const Shape& shape);

private:
    void init(const Shape& shape);

    struct Descriptor;
    std::shared_ptr<Descriptor> m_descriptor;
};

} // namespace deepworks

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);
