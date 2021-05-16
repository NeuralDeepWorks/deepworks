#include <deepworks/parameter.hpp>
#include <deepworks/tensor.hpp>

struct deepworks::Parameter::Impl {
    deepworks::Tensor m_data;
    deepworks::Tensor m_grad;
    bool              m_is_trainable;
};

deepworks::Parameter::Parameter(deepworks::Tensor&& data, bool is_trainable)
    : m_impl(new deepworks::Parameter::Impl()) {
    m_impl->m_data = std::move(data);
    m_impl->m_grad = deepworks::Tensor::zeros(m_impl->m_data.shape());
    m_impl->m_is_trainable = is_trainable;
}

const deepworks::Tensor& deepworks::Parameter::data() const {
    return m_impl->m_data;
}

deepworks::Tensor& deepworks::Parameter::data() {
    return m_impl->m_data;
}

const deepworks::Tensor& deepworks::Parameter::grad() const {
    return m_impl->m_grad;
}

deepworks::Tensor& deepworks::Parameter::grad() {
    return m_impl->m_grad;
}

void deepworks::Parameter::train(bool mode) {
    m_impl->m_is_trainable = mode;
}

bool deepworks::Parameter::is_trainable() const {
    return m_impl->m_is_trainable;
}
