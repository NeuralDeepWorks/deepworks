// FIXME: This file shouldn't be a part of public API !

#pragma once

#include <any>
#include <unordered_map>
#include <memory>

namespace deepworks {

class Attribute {
public:
    Attribute() = default;

    template <typename T>
    Attribute(T&& value) : m_value(std::forward<T>(value)) { }

    template <typename T>
    const T& get() const {
        return *std::any_cast<T>(&m_value);
    }

private:
    std::any m_value;
};

using Attributes = std::unordered_map<std::string, Attribute>;

class LayerInfo {
public:
    LayerInfo() = default;
    LayerInfo(std::string name, std::string type);

    struct Priv {
        std::string           name;
        std::string           type;
        deepworks::Attributes attrs;
     /* deepworks::Parameters params */
    };

    const std::string name()   const { return m_priv->name;     }
    const std::string type()   const { return m_priv->type;     }
 /* const Parameters  params() const { return m_priv->params(); } */

    const Priv& priv() const;
          Priv& priv();
private:
    std::shared_ptr<Priv> m_priv;
};

} // namespace deepworks
