## Benchmark simple neural network architecture on MNIST dataset.

### Architecture:
- Convolution2D(in_channels=3, out_channels=64, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1));
- MaxPooling2D(kernel_size=(2, 2), padding=(0, 0), stride=(2, 2))
- ReLU()

- Convolution2D(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))
- MaxPooling2D(kernel_size=(2, 2), padding=(0, 0), stride=(2, 2))
- ReLU()

- Linear(in_features=4096, out_features=382)
- ReLU()
- BatchNorm1D(eps=0.001, alpha=0.05)

- Linear(in_features=384, out_features=192)
- ReLU()
- BatchNorm1D(eps=0.001, alpha=0.05)

- Linear(in_features=192, out_features=10)
- Softmax()

### Setup
* Follow instruction to setup enviroment: [DeepWorks benchmarks](../README.md)

* Run benchmark:
```bash
cd <path-to-deepworks-root>/build
make benchmark_cifar10
<path-to-deepworks-root>/benchmarks/cifar10/run_cifar10_benchmark.sh
```
