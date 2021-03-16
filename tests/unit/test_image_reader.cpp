#include <fstream>

#include <deepworks/shape.hpp>
#include <deepworks/tensor.hpp>
#include <deepworks/image_reader.hpp>
#include <numeric>

#include "test_utils.hpp"

namespace {
// the reference image data was written in H, W, C loop
deepworks::Tensor GetTensorFromBinary(std::istream &expected_stream, const deepworks::Shape& shape) {
    deepworks::Tensor tensor(shape);
    const size_t total_elements = tensor.total();
    auto it = std::istreambuf_iterator<char>(expected_stream);
    // copy works only when the layout of the reference and tested tensor matches

    auto* tensor_dst = tensor.data();
    for (size_t index = 0; index < total_elements; ++index) {
        tensor_dst[index] = static_cast<uint8_t>(*it);
        ++it;
    }
    return tensor;
}
}

TEST(ImageReader, ReadRGBPng) {
    std::string image_path = deepworks::testutils::GetTestDataPath();
    image_path += "/image/lenna.png";
    std::string reference_path = deepworks::testutils::GetTestDataPath();
    reference_path += "/image/lenna_reference.bin";

    const deepworks::Tensor actual_tensor = deepworks::io::ReadImage(image_path);
    const auto expected_shape = deepworks::Shape{512, 512, 3};

    std::fstream stream(reference_path, std::ios_base::binary | std::ios_base::in);
    auto expected_tensor = GetTensorFromBinary(stream, expected_shape);

    deepworks::testutils::AssertTensorEqual(actual_tensor, expected_tensor);
}

TEST(ImageReader, ReadTransparentPng) {
    std::string image_path = deepworks::testutils::GetTestDataPath();
    image_path += "/image/transparent.png";
    std::string reference_path = deepworks::testutils::GetTestDataPath();
    reference_path += "/image/transparent_reference.bin";

    const deepworks::Tensor actual_tensor = deepworks::io::ReadImage(image_path);
    const auto expected_shape = deepworks::Shape{600, 800, 4};

    std::fstream stream(reference_path, std::ios_base::binary | std::ios_base::in);
    auto expected_tensor = GetTensorFromBinary(stream, expected_shape);

    deepworks::testutils::AssertTensorEqual(actual_tensor, expected_tensor);
}

TEST(ImageReader, ReadRGBJPEG) {
    std::string image_path = deepworks::testutils::GetTestDataPath();
    image_path += "/image/sunset.jpg";
    std::string reference_path = deepworks::testutils::GetTestDataPath();
    reference_path += "/image/sunset_reference.bin";

    const deepworks::Tensor actual_tensor = deepworks::io::ReadImage(image_path);
    const auto expected_shape = deepworks::Shape{600, 800, 3};

    std::fstream stream(reference_path, std::ios_base::binary | std::ios_base::in);
    auto expected_tensor = GetTensorFromBinary(stream, expected_shape);

    deepworks::testutils::AssertTensorEqual(actual_tensor, expected_tensor);
}

TEST(ImageReader, ReadGrayScaleJPEG) {
    std::string image_path = deepworks::testutils::GetTestDataPath();
    image_path += "/image/grayscale.jpg";
    std::string reference_path = deepworks::testutils::GetTestDataPath();
    reference_path += "/image/grayscale_reference.bin";

    const deepworks::Tensor actual_tensor = deepworks::io::ReadImage(image_path);
    const auto expected_shape = deepworks::Shape{600, 600, 1};

    std::fstream stream(reference_path, std::ios_base::binary | std::ios_base::in);
    auto expected_tensor = GetTensorFromBinary(stream, expected_shape);

    deepworks::testutils::AssertTensorEqual(actual_tensor, expected_tensor);
}
