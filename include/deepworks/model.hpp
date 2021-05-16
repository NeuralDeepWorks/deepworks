#pragma once

#include <vector>
#include <memory>

#include <deepworks/placeholder.hpp>
#include <deepworks/layer.hpp>
#include <deepworks/tensor.hpp>
#include <deepworks/parameter.hpp>

namespace deepworks {

class Model {
public:
    struct Config;
    static Model Build(const Config& cfg);

    Model() = default;
    Model(Placeholder  in,  Placeholder  out );
    Model(Placeholders ins, Placeholders outs);

    const Placeholders& inputs()  const;
    const Placeholders& outputs() const;
    const Layers      & layers()  const;
          Layers      & layers();

    Layer getLayer(const std::string& name);
    ParamMap& params();

    using StateDict = std::unordered_map<std::string, Tensor>;
    const StateDict& state() const;
          StateDict& state();

    const Config& cfg() const;

    void train(bool mode);

    // Execution API
    void compile();
    void forward(const Tensor& input, Tensor& outputs);
    void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);

    void backward(const Tensor& input,
                  const Tensor& output,
                  const Tensor& grad_output);

    void backward(const std::vector<Tensor>& inputs,
                  const std::vector<Tensor>& outputs,
                  const std::vector<Tensor>& grad_outputs);

private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

struct Model::Config {
    struct OperationInfo;
    struct PlaceholderInfo;

    using PhInfoIdMap = std::unordered_map<int, PlaceholderInfo>;
    using PhInfos     = std::vector<PlaceholderInfo>;
    using OpInfos     = std::vector<OperationInfo>;

    struct PlaceholderInfo {
        Shape shape;
        int   id;
    };

    struct OperationInfo {
        std::string name;
        std::string type;
        Attributes  attrs;

        std::vector<int> in_ids;
        std::vector<int> out_ids;
    };

    std::vector<int> input_ids;
    std::vector<int> output_ids;
    PhInfoIdMap      ph_map;
    OpInfos          sorted_ops;
};

} // namespace deepworks
