#pragma once

#include <deepworks/model.hpp>
#include <iostream>

namespace deepworks {
    void save(const Model::StateDict& state, const std::string& path);
    void load(      Model::StateDict& state, const std::string& path);
}
