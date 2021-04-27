#pragma once

#include <deepworks/tensor.hpp>

#include "runtime/cpu/layers/cpulayer.hpp"
#include "runtime/cpu/kernels/kernels.hpp"

#include "util/assert.hpp"

namespace deepworks {
namespace cpu {

class CPUBatchNorm1D : public ICPULayer {
public:
    CPUBatchNorm1D(deepworks::LayerInfo&& info);

    virtual void forward(const std::vector<Tensor>& inputs,
                               std::vector<Tensor>& outputs) override;

    virtual void updateGradients(const std::vector<Tensor>& inputs,
                                 const std::vector<Tensor>& grad_outputs) override;

    virtual void backward(const std::vector<Tensor>& inputs,
                          const std::vector<Tensor>& outputs,
                          const std::vector<Tensor>& grad_outputs,
                          std::vector<Tensor>& grad_inputs) override;
private:
    void validate(const std::vector<Tensor>& inputs,
                  std::vector<Tensor>& outputs);


    deepworks::Tensor m_gamma, m_beta;
    deepworks::Tensor m_grad_gamma, m_grad_beta;
    deepworks::Tensor m_std, m_running_mean, m_running_var;
    deepworks::Tensor m_input_centered;
};

class CPUBatchNorm2D : public ICPULayer {
public:
    CPUBatchNorm2D(deepworks::LayerInfo&& info);

    virtual void forward(const std::vector<Tensor>& inputs,
                               std::vector<Tensor>& outputs) override;

    virtual void updateGradients(const std::vector<Tensor>& inputs,
                                 const std::vector<Tensor>& grad_outputs) override;

    virtual void backward(const std::vector<Tensor>& inputs,
                          const std::vector<Tensor>& outputs,
                          const std::vector<Tensor>& grad_outputs,
                          std::vector<Tensor>& grad_inputs) override;
private:
    void validate(const std::vector<Tensor>& inputs,
                  std::vector<Tensor>& outputs);


    deepworks::Tensor m_gamma, m_beta;
    deepworks::Tensor m_std, m_running_mean, m_running_var;
    deepworks::Tensor m_grad_gamma, m_grad_beta;
    deepworks::Tensor m_input_centered;
    deepworks::Tensor nhwc_in, nhwc_out;
};

} // namespace cpu
} // namespace deepworks
