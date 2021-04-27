#include <fstream>

#include <deepworks/serialization.hpp>

#include "util/assert.hpp"

void deepworks::save(const Model::StateDict& state, const std::string& path) {
    std::ofstream file(path, std::ios::out | std::ios::binary);

    size_t size = state.size();
    file.write(reinterpret_cast<char*>(&size), sizeof(size));
    
    for (const auto& [name, tensor] : state) {
        size_t size_str = name.size();
        file.write(reinterpret_cast<char*>(&size_str), sizeof(size_str));
        file.write(&name[0], size_str);

        size_t shape_size = tensor.shape().size();
        file.write(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
        file.write(reinterpret_cast<const char*>(tensor.shape().data()),
                   tensor.shape().size() * sizeof(deepworks::Shape::value_type));
        file.write(reinterpret_cast<char*>(tensor.data()),
                   tensor.total() * sizeof(deepworks::Tensor::Type));
    }
}

void deepworks::load(Model::StateDict& state, const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        DeepWorks_Throw() << "Failed to open the file: " << path;
    }

    size_t size = 0u;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    for (int i = 0; i < size; ++i) {
        size_t size_str = 0;
        std::string name;
        file.read(reinterpret_cast<char*>(&size_str), sizeof(size_str));
        name.resize(size_str);
        file.read(&name[0], size_str);

        auto it = state.find(name);
        if (it == state.end()) {
            DeepWorks_Throw() << "Failed to find: " << name << " in state dict ";
        }

        size_t shape_size = 0u;
        deepworks::Shape shape;
        file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
        shape.resize(shape_size);
        file.read(reinterpret_cast<char*>(shape.data()),
                   shape_size * sizeof(deepworks::Shape::value_type));

        if (it->second.shape() != shape) {
            DeepWorks_Throw() << "Shape mismatch for: " << name << "\n"
                              << "Expected: " << it->second.shape()
                              << "Actual: " << shape;
        }

        file.read(reinterpret_cast<char*>(it->second.data()),
                  it->second.total() * sizeof(deepworks::Tensor::Type));
    }
}
