#pragma once

#include <memory> // shared_ptr

#include <deepworks/shape.hpp>

namespace deepworks {

class Call;
class Placeholder {
public:
    explicit Placeholder(const deepworks::Shape& shape);

    const Shape& shape() const;

    // NB: Public for test, but not available for user,
    // because Impl isn't exported to public API.
    Placeholder(const deepworks::Shape& shape, Call call);
    struct Impl;
    const Impl& impl() const;
    std::shared_ptr<Impl> m_impl;
};

} // namespace deepworks
