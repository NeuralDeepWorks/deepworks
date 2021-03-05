#include <deepworks/tensor.hpp>

#include "model/graph.hpp"
#include "runtime/backend.hpp"
#include "runtime/cpu/layers/cpulayer.hpp"

namespace deepworks {

class CPUBackend : public IBackend {
public:
    CPUBackend(graph::Graph& g, graph::TypedGraph& tg, int batch_size);

    std::vector<Tensor> forward(const std::vector<Tensor>& tensors)  override;
    std::vector<Tensor> backward(const std::vector<Tensor>& tensors) override;

private:
    graph::Graph&      m_g;
    graph::TypedGraph& m_tg;
    int                m_bs;

    std::vector<Tensor> m_mem;
    std::vector<deepworks::cpu::ICPULayer::Ptr> m_layers;
};

} // namespace deepworks
