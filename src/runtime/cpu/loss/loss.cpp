#include <vector>

#include "loss.hpp"

namespace deepworks {

void deepworks::CrossEntropyLoss::CPUCrossEntropyLossForward(const ConstMatrix &X,
                                                             const ConstVector &target,
                                                             Matrix &loss) {
    int rows = X.rows();
    int cols = X.cols();

    Eigen::MatrixXf pr(rows, cols);
    Matrix P(pr.data(), rows, cols);

    CPUSoftmaxForward(X, P);

    Vector V(pr.data(), pr.size());

    std::vector<int> slice(rows);

    for (int i = 0; i < rows; ++i) {
        slice[i] = static_cast<int>(target(0, i)) + cols * i;
    }

    loss(0, 0) = -V(0, slice).array().log().sum() / static_cast<float>(rows);
}


void deepworks::CrossEntropyLoss::CPUCrossEntropyLossBackward(const ConstMatrix &X,
                                                              const ConstVector &target,
                                                              Matrix &grad_output) {
    int rows = X.rows();
    int cols = X.cols();

    CPUSoftmaxForward(X, grad_output);

    Vector V(grad_output.data(), grad_output.size());

    std::vector<int> slice(rows);

    for (int i = 0; i < rows; ++i) {
        slice[i] = static_cast<int>(target(0, i)) + cols * i;
    }

    V(0, slice).array() -= 1.0;

    grad_output /= static_cast<float>(rows);
}

} // deepworks
