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
        fclose(infile);
        DeepWorks_Throw() << "can't open file: " << path;
    }

    int width         = -1;
    int height        = -1;
    int channels      = -1;
    JSAMPARRAY buffer = nullptr;

    auto destroy = [&]() {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
    };

    auto finish_and_destroy = [&]() {
        (void) jpeg_finish_decompress(&cinfo);
        destroy();
    };

    try {
        cinfo.err = jpeg_std_error(&err);
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, infile);

        (void) jpeg_read_header(&cinfo, true);
        (void) jpeg_start_decompress(&cinfo);

        size_t row_stride = cinfo.output_width * cinfo.output_components;
        buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

        width = static_cast<int>(cinfo.output_width);
        height = static_cast<int>(cinfo.output_height);
        channels = static_cast<int>(cinfo.output_components);

        if (out_tensor.empty()) {
            out_tensor.allocate(deepworks::Shape{height, width, channels});
        } else {
            DeepWorks_Assert(height == out_tensor.shape()[0]);
            DeepWorks_Assert(width == out_tensor.shape()[1]);
            DeepWorks_Assert(channels == out_tensor.shape()[2]);
        }
    } catch (...) {
        // NB: jpeg_finish_decompress cause "Application transferred to few scanlines"
        destroy();
        throw;
    }

    deepworks::Tensor::Type *dst_data = out_tensor.data();
    deepworks::Strides tensor_strides = out_tensor.strides();

    const size_t elements_per_h_channel = width * channels;
    size_t h = 0;
    // it's works if we have default hwc layout for tensor
    try {
        while (cinfo.output_scanline < cinfo.output_height) {
            (void) jpeg_read_scanlines(&cinfo, buffer, 1);
            std::copy_n(buffer[0], elements_per_h_channel, &dst_data[h * tensor_strides[0]]);
            ++h;
        }
    } catch (...) {
        finish_and_destroy();
        throw;
    }
    finish_and_destroy();

#else
    DeepWorks_Assert(false && "Couldn't find LIBJPEG");
#endif
}

void ReadPngFile(std::string_view path, deepworks::Tensor& out_tensor) {
#ifdef HAVE_PNG
    FILE *infile = fopen(path.data(), "rb");
    if (!infile) {
        fclose(infile);
        DeepWorks_Throw() << "can't open file: " << path;
    }

    png_bytep*  row_pointers = nullptr;
    png_structp png_ptr{};
    png_infop   info_ptr{};
    try {
        char header[8];
        // FIXME: Handle that properly.
        size_t sz = fread(header, 1, 8, infile);
        (void)sz;
        if (png_sig_cmp(reinterpret_cast<png_const_bytep>(header), 0, 8)) {
            DeepWorks_Throw() << "File is not a PNG file: " << path;
        }
        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        info_ptr = png_create_info_struct(png_ptr);

        DeepWorks_Assert(!setjmp(png_jmpbuf(png_ptr)) && "Error during init_io");

        png_init_io(png_ptr, infile);
        png_set_sig_bytes(png_ptr, 8);

        png_read_info(png_ptr, info_ptr);

        int width = static_cast<int>(png_get_image_width(png_ptr, info_ptr));
        int height = static_cast<int>(png_get_image_height(png_ptr, info_ptr));
        int channels = static_cast<int>(png_get_channels(png_ptr, info_ptr));

        png_read_update_info(png_ptr, info_ptr);
        if (out_tensor.empty()) {
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

        row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);
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
    } catch (...) {
        free(row_pointers);
        fclose(infile);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        throw;
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
void ReadImage(std::string_view path, Tensor& tensor) {
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
    ReadImage(path, out);
    return out;
}
}
