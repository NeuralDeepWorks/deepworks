#include <fstream>

#include <deepworks/serialization.hpp>
#include <ade/util/zip_range.hpp>

#include "util/assert.hpp"
#include <iostream>

namespace dw = deepworks;

template <typename T>
void write_value(T value, std::ofstream& file) {
    file.write(reinterpret_cast<char*>(&value), sizeof(value));
}

template <typename T>
T read_value(std::ifstream& file) {
    T value{};
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

template <typename T>
void write_sequence(const T& seq, std::ofstream& file) {
    write_value(seq.size(), file);
    file.write(reinterpret_cast<const char*>(seq.data()),
               seq.size() * sizeof(typename T::value_type));
}

template <typename T>
T read_sequence(std::ifstream& file) {
    T seq{};
    auto size = read_value<size_t>(file);
    seq.resize(size);
    file.read(reinterpret_cast<char*>(seq.data()),
              size * sizeof(typename T::value_type));
    return seq;
}

// NB: WA for reading array
template <typename T, int S>
std::array<T, S> read_array(std::ifstream& file) {
    std::array<T, S> seq;
    auto size = read_value<size_t>(file);
    file.read(reinterpret_cast<char*>(seq.data()),
              size * sizeof(T));
    return seq;
}

enum class SerializationKind : int {
    STATE,
    CONFIG
};

void dw::save_state(const Model::StateDict& state, const std::string& path) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    dw::save_state(state, file);
}

void dw::save_state(const Model::StateDict& state, std::ofstream& file) {
    write_value(static_cast<int>(SerializationKind::STATE), file);
    write_value(state.size(), file);
    
    for (const auto& [name, tensor] : state) {
        write_sequence(name          , file);
        write_sequence(tensor.shape(), file);
        file.write(reinterpret_cast<char*>(tensor.data()),
                   tensor.total() * sizeof(dw::Tensor::Type));
    }
}

void dw::load_state(Model::StateDict& state, const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        DeepWorks_Throw() << "Failed to open the file: " << path;
    }
    dw::load_state(state, file);
}

void dw::load_state(Model::StateDict& state, std::ifstream& file) {
    auto kind = static_cast<SerializationKind>(read_value<int>(file));
    if (kind != SerializationKind::STATE) {
        DeepWorks_Throw() << "File doesn't contain state dict";
    }

    auto size = read_value<size_t>(file);
    
    for (int i = 0; i < size; ++i) {
        auto name = read_sequence<std::string>(file);
        auto it = state.find(name);
        if (it == state.end()) {
            DeepWorks_Throw() << "Failed to find: " << name << " in state dict ";
        }

        auto shape = read_sequence<dw::Shape>(file);
        if (it->second.shape() != shape) {
            DeepWorks_Throw() << "Shape mismatch for: " << name << "\n"
                              << "Expected: " << it->second.shape()
                              << "Actual: " << shape;
        }

        file.read(reinterpret_cast<char*>(it->second.data()),
                  it->second.total() * sizeof(dw::Tensor::Type));
    }
}

void dw::save_dot(const dw::Model::Config& cfg,
                  const std::string& path) {
    std::ofstream file(path, std::ios::out);

    file << "digraph Deepworks_model {\n";

    for (const auto& [id, ph] : cfg.ph_map) {
        std::string name = "Placeholder_" + std::to_string(ph.id);
        file << name << " ";
        file << "[label=\"" << name << "\n" << ph.shape << "\" ]" << std::endl;
    }

    for (const auto& op : cfg.sorted_ops) {
        // FIXME: Dump more info, attributes, parameters, named buffers...
        file << op.name << " [shape=box, label=\"" << op.name
                        << " (" << op.type << ")" << "\" ]" << std::endl;
    }

    for (const auto& op : cfg.sorted_ops) {
        for (auto it : ade::util::indexed(op.in_ids)) {
            auto port  = ade::util::index(it);
            auto in_id = ade::util::value(it);

            const auto& ph = cfg.ph_map.at(in_id);
            std::string name = "Placeholder_" + std::to_string(ph.id);
            file << name << " -> \"" << op.name << "\" [ label = \"in_port: "
               << port << "\" ]" << std::endl;
        }

        for (auto it : ade::util::indexed(op.out_ids)) {
            auto port   = ade::util::index(it);
            auto out_id = ade::util::value(it);

            const auto& ph = cfg.ph_map.at(out_id);
            std::string name = "Placeholder_" + std::to_string(ph.id);
            file << op.name << " -> " << name << " [ label = \"out_port: "
               << port << "\" ];" << std::endl;
        }
    }

    file << "}";
}

void write_attribute(const dw::Attribute& attr, std::ofstream& file) {
    write_value(static_cast<int>(attr.shape), file);
    write_value(static_cast<int>(attr.type) , file);
    switch (attr.shape) {
        case dw::AttrShape::VALUE: {
           switch (attr.type) {
               case dw::AttrType::INT:
                   write_value(attr.get<int>(), file);
                   break;
               case dw::AttrType::FLOAT:
                   write_value(attr.get<float>(), file);
                   break;
               default: DeepWorks_Assert(false && "Unsupported attribute type");
           }
           break;
        }
        case dw::AttrShape::ARRAY2: {
           switch (attr.type) {
               case dw::AttrType::INT:
                   write_sequence(attr.get<std::array<int, 2>>(), file);
                   break;
               case dw::AttrType::FLOAT:
                   write_sequence(attr.get<std::array<float, 2>>(), file);
                   break;
               default: DeepWorks_Assert(false && "Unsupported attribute type");
           }
           break;
        }
        default: DeepWorks_Assert(false && "Unsupported attribute shape");
    }
}

dw::Attribute read_attribute(std::ifstream& file) {
    auto shape = static_cast<dw::AttrShape>(read_value<int>(file));
    auto type  = static_cast<dw::AttrType> (read_value<int>(file));
    switch (shape) {
        case dw::AttrShape::VALUE: {
           switch (type) {
               case dw::AttrType::INT:
                   return dw::Attribute(read_value<int>(file));
               case dw::AttrType::FLOAT:
                   return dw::Attribute(read_value<float>(file));
               default: DeepWorks_Assert(false && "Unsupported attribute type");
           }
           break;
        }
        case dw::AttrShape::ARRAY2: {
           switch (type) {
               case dw::AttrType::INT:
                   return dw::Attribute(read_array<int, 2>(file));
               case dw::AttrType::FLOAT:
                   return dw::Attribute(read_array<float, 2>(file));
               default: DeepWorks_Assert(false && "Unsupported attribute type");
           }
           break;
        }
        default: DeepWorks_Assert(false && "Unsupported attribute shape");
    }
    // NB: Unreachable code
    DeepWorks_Assert(false);
}

void write_attributes(const dw::Attributes& attrs, std::ofstream& file) {
    write_value(attrs.size(), file);
    for (const auto& [name, attr] : attrs) {
        write_sequence (name, file);
        write_attribute(attr, file);
    }
}

dw::Attributes read_attributes(std::ifstream& file) {
    dw::Attributes attrs;
    auto attrs_size = read_value<size_t>(file);
    for (int i = 0; i < attrs_size; ++i) {
        auto name = read_sequence<std::string>(file);
        auto attr = read_attribute(file);
        attrs.emplace(std::move(name), std::move(attr));
    }
    return attrs;
}

void dw::save_cfg(const Model::Config& cfg, const std::string& path) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    dw::save_cfg(cfg, file);
}

void dw::save_cfg(const Model::Config& cfg, std::ofstream& file) {
    write_value(static_cast<int>(SerializationKind::CONFIG), file);
    // NB: Save in/out ids
    write_sequence(cfg.input_ids, file);
    write_sequence(cfg.output_ids, file);

    // NB: Save ph_map
    write_value(cfg.ph_map.size(), file);
    for (const auto& [id, ph_info] : cfg.ph_map) {
        // NB: Save PlaceholderInfo
        write_value(id, file);
        write_sequence(ph_info.shape, file);
    }

    // NB: Save sorted operations
    write_value(cfg.sorted_ops.size(), file);
    for (const auto& op_info : cfg.sorted_ops) {
        write_sequence(op_info.name   , file);
        write_sequence(op_info.type   , file);
        write_sequence(op_info.in_ids , file);
        write_sequence(op_info.out_ids, file);
        write_attributes(op_info.attrs, file);
    }
}

dw::Model::Config dw::load_cfg(const std::string& path) {
    std::ifstream file(path, std::ios::out | std::ios::binary);
    return dw::load_cfg(file);
}

dw::Model::Config dw::load_cfg(std::ifstream& file) {
    auto kind = static_cast<SerializationKind>(read_value<int>(file));
    if (kind != SerializationKind::CONFIG) {
        DeepWorks_Throw() << "File doesn't contain config";
    }

    dw::Model::Config cfg;
    cfg.input_ids  = read_sequence<std::vector<int>>(file);
    cfg.output_ids = read_sequence<std::vector<int>>(file);

    auto map_size = read_value<size_t>(file);
    for (int i = 0; i < map_size; ++i) {
        auto id    = read_value<int>(file);
        auto shape = read_sequence<dw::Shape>(file);
        cfg.ph_map.emplace(id,
                dw::Model::Config::PlaceholderInfo{shape, id});
    }

    auto num_ops = read_value<size_t>(file);
    for (int i = 0; i < num_ops; ++i) {
        dw::Model::Config::OperationInfo op_info;
        op_info.name    = read_sequence<std::string>(file);
        op_info.type    = read_sequence<std::string>(file);
        op_info.in_ids  = read_sequence<std::vector<int>>(file);
        op_info.out_ids = read_sequence<std::vector<int>>(file);
        op_info.attrs   = read_attributes(file);
        cfg.sorted_ops.push_back(std::move(op_info));
    }

    return cfg;
}

void dw::save(const dw::Model& model, const std::string& path) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    dw::save_cfg(model.cfg()  , file);
    dw::save_state(model.state(), file);
}

dw::Model dw::load(const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    auto model = dw::Model::Build(dw::load_cfg(file));
    dw::load_state(model.state(), file);
    return model;
}
