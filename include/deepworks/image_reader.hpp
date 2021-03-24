#pragma once

#include <string>
#include <optional>

namespace deepworks {

class Tensor;

namespace io {
deepworks::Tensor ReadImage(std::string_view);

void ReadImageToTensor(std::string_view path, Tensor&);
} // namespace io

} // namespace deepworks
