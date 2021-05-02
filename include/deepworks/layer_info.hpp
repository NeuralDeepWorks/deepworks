// FIXME: This file shouldn't be a part of public API !

#pragma once

#include <any>
#include <unordered_map>
#include <memory>

#include <deepworks/parameter.hpp>
#include <deepworks/tensor.hpp>

namespace deepworks {

enum class AttrType : int {
    UNSUPPORTED,
    INT,
    FLOAT
};

enum class AttrShape : int {
    VALUE,
    ARRAY2,
};

template <typename T>
struct AttrTypeTraits {
    static constexpr const AttrType type = AttrType::UNSUPPORTED;
};

template <>
struct AttrTypeTraits<int> {
    static constexpr const AttrType type = AttrType::INT;
};

template <>
struct AttrTypeTraits<float> {
    static constexpr const AttrType type = AttrType::FLOAT;
};

template <typename T>
struct AttrTraits {
    static constexpr const AttrType  type  = AttrTypeTraits<T>::type;
    static constexpr const AttrShape shape = AttrShape::VALUE;
};

template <typename U>
struct AttrTraits<std::array<U, 2>> {
    static constexpr const AttrType  type  = AttrTypeTraits<U>::type;
    static constexpr const AttrShape shape = AttrShape::ARRAY2;
};

class Attribute {
public:
    Attribute() = default;

    template <typename T>
    Attribute(T&& value)
        : m_value(std::forward<T>(value)),
          m_type (AttrTraits<typename std::decay<T>::type>::type),
          m_shape(AttrTraits<typename std::decay<T>::type>::shape) {
    }

    template <typename T>
    const T& get() const {
        return *std::any_cast<T>(&m_value);
    }

    AttrType type() const {
        return m_type;
    }

    AttrShape shape() const {
        return m_shape;
    }

private:
    AttrType  m_type;
    AttrShape m_shape;
    std::any  m_value;
};

using Attributes = std::unordered_map<std::string, Attribute>;

using BufferMap = std::unordered_map<std::string, Tensor>;
class LayerInfo {
public:
    LayerInfo() = default;
    LayerInfo(std::string name, std::string type);

    struct Impl {
        std::string           name;
        std::string           type;
        deepworks::Attributes attrs;
        deepworks::ParamMap   params;
        deepworks::BufferMap  buffers;
    };

    const std::string name()    const { return m_impl->name;    }
    const std::string type()    const { return m_impl->type;    }
    const ParamMap&   params()  const { return m_impl->params;  }
    const BufferMap&  buffers() const { return m_impl->buffers; }

    const Impl& impl() const;
          Impl& impl();
private:
    std::shared_ptr<Impl> m_impl;
};

} // namespace deepworks
