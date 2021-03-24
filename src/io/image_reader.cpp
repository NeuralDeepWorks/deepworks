#include <stdexcept>
#include <sstream>

#include <deepworks/tensor.hpp>
#include "util/assert.hpp"

#ifdef HAVE_JPEG
#include <jpeglib.h>
#endif

#ifdef HAVE_PNG
#include <png.h>
#include <algorithm>

#endif

namespace {
bool IsPngFile(std::string_view path) {
    return path.substr(path.find_last_of(".") + 1) == "png";
}

bool IsJpegFile(std::string_view path) {
    return path.substr(path.find_last_of(".") + 1) == "jpg" || path.substr(path.find_last_of(".") + 1) == "jpeg";
}

void ReadJpegFile(std::string_view path, deepworks::Tensor& out_tensor) {
#ifdef HAVE_JPEG
    struct jpeg_decompress_struct cinfo{};
    struct jpeg_error_mgr err{};
    FILE *infile = fopen(path.data(), "rb");

    if (!infile) {
        std::stringstream fmt;
        fmt << "can't open file: " << path;
        DeepWorks_Assert(false && fmt.str().c_str());
    }

    cinfo.err = jpeg_std_error(&err);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    (void) jpeg_read_header(&cinfo, true);
    (void) jpeg_start_decompress(&cinfo);

    size_t row_stride = cinfo.output_width * cinfo.output_components;
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    int width = static_cast<int>(cinfo.output_width);
    int height = static_cast<int>(cinfo.output_height);
    int channels = static_cast<int>(cinfo.output_components);

    if (out_tensor.data() == nullptr) {
        out_tensor.allocate(deepworks::Shape{height, width, channels});
    } else {
        DeepWorks_Assert(height == out_tensor.shape()[0]);
        DeepWorks_Assert(width == out_tensor.shape()[1]);
        DeepWorks_Assert(channels == out_tensor.shape()[2]);
    }

    deepworks::Tensor::Type *dst_data = out_tensor.data();
    deepworks::Strides tensor_strides = out_tensor.strides();

    const size_t elements_per_h_channel = width * channels;
    size_t h = 0;
    // it's works if we have default hwc layout for tensor
    while (cinfo.output_scanline < cinfo.output_height) {
        (void) jpeg_read_scanlines(&cinfo, buffer, 1);
        std::copy_n(buffer[0], elements_per_h_channel, &dst_data[h * tensor_strides[0]]);
        ++h;
    }

    (void) jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
#else
    DeepWorks_Assert(false && "Couldn't find LIBJPEG");
#endif
}

void ReadPngFile(std::string_view path, deepworks::Tensor& out_tensor) {
#ifdef HAVE_PNG
    FILE *infile = fopen(path.data(), "rb");
    if (!infile) {
        std::stringstream fmt;
        fmt << "can't open file: " << path;
        DeepWorks_Assert(false && fmt.str().c_str());
    }
    char header[8];
    fread(header, 1, 8, infile);
    if (png_sig_cmp(reinterpret_cast<png_const_bytep>(header), 0, 8)) {
        std::stringstream fmt;
        fmt << "File is not a PNG file: " << path;
        DeepWorks_Assert(false && fmt.str().c_str());
    }
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    png_infop info_ptr = png_create_info_struct(png_ptr);

    DeepWorks_Assert(!setjmp(png_jmpbuf(png_ptr)) && "Error during init_io");

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    int width = static_cast<int>(png_get_image_width(png_ptr, info_ptr));
    int height = static_cast<int>(png_get_image_height(png_ptr, info_ptr));
    int channels = static_cast<int>(png_get_channels(png_ptr, info_ptr));

    png_read_update_info(png_ptr, info_ptr);
    if (out_tensor.data() == nullptr) {
        out_tensor.allocate(deepworks::Shape{height, width, channels});
    } else {
        DeepWorks_Assert(height == out_tensor.shape()[0]);
        DeepWorks_Assert(width == out_tensor.shape()[1]);
        DeepWorks_Assert(channels == out_tensor.shape()[2]);
    }

    deepworks::Tensor::Type *dst_data = out_tensor.data();
    deepworks::Strides tensor_strides = out_tensor.strides();
    /* read file */
    DeepWorks_Assert(!setjmp(png_jmpbuf(png_ptr)) && "Error during read image");

    auto *row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte *) malloc(png_get_rowbytes(png_ptr, info_ptr));
    }
    png_read_image(png_ptr, row_pointers);
    // it's works if we have default hwc layout for tensor
    const size_t elements_per_h_channel = width * channels;
    for (int h = 0; h < height; ++h) {
        std::copy_n(row_pointers[h], elements_per_h_channel, &dst_data[h * tensor_strides[0]]);
    }
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    fclose(infile);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
#else
    DeepWorks_Assert(false && "Couldn't find LIBPNG");
#endif
}
}

namespace deepworks::io {
void ReadImageToTensor(std::string_view path, Tensor& tensor) {
    if (IsPngFile(path)) {
        ReadPngFile(path, tensor);
        return;
    }
    if (IsJpegFile(path)) {
        ReadJpegFile(path, tensor);
        return;
    }
    DeepWorks_Assert(false && "image format not supported");
}

deepworks::Tensor ReadImage(std::string_view path) {
    Tensor out;
    ReadImageToTensor(path, out);
    return out;
}
}
