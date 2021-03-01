#pragma once

#include <vector>
#include <memory>

namespace deepworks {

using Shape = std::vector<int>;
using Strides = std::vector<size_t>;

class Tensor {
public:
    using Type = float;

    Tensor();
    explicit Tensor(const Shape &shape);

    void copyTo(Tensor &tensor);
    Type *data();
    size_t total() const;
    void create();
    bool isCreated() const;
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
    size_t total();
    void create();

    ~Descriptor();

    Strides m_strides;
    Shape m_shape;
    Type *m_data{nullptr};
    bool m_created{false};
};

} // namespace deepworks
