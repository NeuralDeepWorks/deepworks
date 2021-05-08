## Benchmark simple neural network architecture on MNIST dataset.

### Architecture:
* Linear(in_features=784, out_features=100)
* ReLU()
* BatchNorm1D(epsilon=0.001, momentum=0.05)
* Linear(in_features=100, out_features=10)
* Softmax()

### Setup
* Follow instruction to setup enviroment: [DeepWorks benchmarks](../README.md)

* Run benchmark:
```bash
cd <path-to-deepworks-root>/build
make benchmark_mnist
<path-to-deepworks-root>/benchmarks/mnist/run_mnist_benchmark.sh
```
