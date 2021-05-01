#pragma once

#include <fstream>
#include <deepworks/model.hpp>

namespace deepworks {
    void save_state(const Model::StateDict& state, std::ofstream&     file);
    void load_state(      Model::StateDict& state, std::ifstream&     file);
    void save_state(const Model::StateDict& state, const std::string& path);
    void load_state(      Model::StateDict& state, const std::string& path);

    void save_dot(const Model::Config& cfg, const std::string& path);

    void save_cfg(const Model::Config& cfg, const std::string& path);
    void save_cfg(const Model::Config& cfg, std::ofstream& file);
    Model::Config load_cfg(const std::string& path);
    Model::Config load_cfg(std::ifstream& file);

    void  save(const Model& model, const std::string& path);
    Model load(const std::string& path);
}
