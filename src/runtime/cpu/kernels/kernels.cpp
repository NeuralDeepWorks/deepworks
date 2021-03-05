#include "runtime/cpu/kernels/kernels.hpp"

void deepworks::cpu::kernels::CPUReLUForward(const MatrixMapper& X, MatrixMapper& result) {
    result = (X.array() > 0).select(X, 0);
}
