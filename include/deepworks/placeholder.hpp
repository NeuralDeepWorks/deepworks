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
    // because Priv isn't exported to public API.
    Placeholder(const deepworks::Shape& shape, Call call);
    struct Priv;
    const Priv& priv() const;
    std::shared_ptr<Priv> m_priv;
};

} // namespace deepworks
