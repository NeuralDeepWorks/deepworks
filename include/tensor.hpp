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
    size_t totalElements() const;
    const Strides &strides() const;
    const Shape &shape() const;

    ~Tensor();

private:
    struct Descriptor;
    std::shared_ptr<Descriptor> _descriptor;
};

} // namespace deepworks
