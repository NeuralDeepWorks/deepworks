#pragma once

#include <string>

namespace deepworks {

class Tensor;

namespace io {
deepworks::Tensor ReadImage(std::string_view);

void ReadImage(std::string_view path, Tensor&);
} // namespace io

} // namespace deepworks
