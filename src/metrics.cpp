#include <numeric>
#include <algorithm>

#include <Eigen/Core>
#include <deepworks/metrics.hpp>
#include "util/assert.hpp"


using ConstMatrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using ConstVector = Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>;

float deepworks::metric::accuracy(const deepworks::Tensor& y_pred,
                                  const deepworks::Tensor& y_true) {
    const auto& shape = y_pred.shape();
    DeepWorks_Assert(shape.size() == 2);
    int rows = shape[0];
    int cols = shape[1];
    DeepWorks_Assert(y_true.total() == rows);

    ConstMatrix y_pred_mat(y_pred.data(), rows, cols);
    ConstVector y_true_vec(y_true.data(), rows);

    float acc = 0;
    ConstMatrix::Index pred_col;
    // FIXME: Calculate without loop
    for (int i = 0; i < rows; i++) {
        y_pred_mat.row(i).maxCoeff(&pred_col);
        auto label = y_true_vec[i];
        acc += pred_col == label;
    }
    return acc / rows;
}

float deepworks::metric::accuracyOneHot(const deepworks::Tensor& y_pred,
                                        const deepworks::Tensor& y_true) {
    const auto& shape = y_pred.shape();
    DeepWorks_Assert(y_true.shape() == shape);
    DeepWorks_Assert(shape.size() == 2);
    int rows = shape[0];
    int cols = shape[1];

    ConstMatrix y_pred_mat(y_pred.data(), rows, cols);
    ConstMatrix y_true_mat(y_true.data(), rows, cols);
    float acc = 0;
    ConstMatrix::Index pred_col, label_col;
    // FIXME: Calculate without loop
    for (int i = 0; i < rows; i++) {
        y_pred_mat.row(i).maxCoeff(&pred_col);
        y_true_mat.row(i).maxCoeff(&label_col);
        acc += pred_col == label_col;
    }
    return acc / rows;
}
