#pragma once

#include <string>

namespace deepworks {

class Tensor;

namespace io {
deepworks::Tensor ReadImage(std::string_view);
} // namespace io

} // namespace deepworks
