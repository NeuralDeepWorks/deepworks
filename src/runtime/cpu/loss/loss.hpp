#include "runtime/cpu/kernels/kernels.hpp"

namespace deepworks {

/*
 * Realization of CrossEntropyLoss
 * This criterion combines LogSoftmax and NLLLoss in one single class
 */
class CrossEntropyLoss {
public:
    CrossEntropyLoss() = default;

    /*
     * CPUCrossEntropyLossForward
     * implements computation of cross entropy loss
     * X have size [batch_size, N_classes]
     * target have size [1, batch_size], where values are in the range [0, C-1]
     * loss have size [1, 1]
     */
    void CPUCrossEntropyLossForward(const ConstMatrix &X, const ConstVector &target, Matrix &loss);

    /*
     * CPUCrossEntropyLossBackward
     * Implements computation backward pass cross entropy loss
     * X have size [batch_size, N_classes]
     * target have size [1, batch_size], where values are in the range [0, C-1]
     * grad_output have size [batch_size, N_classes]
     */
    void CPUCrossEntropyLossBackward(const ConstMatrix &X, const ConstVector &target, Matrix &grad_output);
};

} // deepworks
