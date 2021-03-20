#pragma once

#include <deepworks/tensor.hpp>

namespace deepworks {
namespace loss {

class CrossEntropyLoss {
public:

    /*
    * CPUCrossEntropyLossForward
    * Implements computation of CrossEntropyLoss
    * criterion combines Log and NLLLoss, applies after Softmax layer
    * X have size [batch_size, N_classes]
    * target have size [1, batch_size], where values are in the range [0, N_classes-1]
    * loss is a float scalar
    */
    float CPUForward(const Tensor& predictions, const Tensor& target);

    /*
    * CPUCrossEntropyLossBackward
    * Implements computation backward pass CrossEntropyLoss
    * X have size [batch_size, N_classes]
    * target have size [1, batch_size], where values are in the range [0, N_classes-1]
    * grad_output have size [batch_size, N_classes]
    */
    void CPUBackward(const Tensor& predictions, const Tensor& target, Tensor& grad_output);

private:
    Tensor log_predictions;
};

} // namespace loss
} // namespace deepworks
