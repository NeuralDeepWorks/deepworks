#pragma once

#include <vector>
#include <memory>
#include <unordered_map>

#include <deepworks/shape.hpp>

namespace deepworks {

using Strides = std::vector<size_t>;

class Tensor {
public:
    using Type = float;

    static Tensor zeros        (const Shape& shape);
    static Tensor constant     (const Shape& shape, float value);
    static Tensor xavierUniform(const Shape& shape);
    static Tensor uniform      (const Shape& shape, float lower = 0.f, float upper = 1.f);

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
    void reshape(const Shape& shape);

private:
    void init(const Shape& shape);

    struct Descriptor;
    std::shared_ptr<Descriptor> m_descriptor;
};

using TensorMap = std::unordered_map<std::string, Tensor>;

} // namespace deepworks

std::ostream& operator<<(std::ostream& stream, const deepworks::Tensor& tensor);
