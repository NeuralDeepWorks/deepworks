# deepworks

[![Build Status](https://travis-ci.org/NeuralDeepWorks/deepworks.svg?branch=main)](https://travis-ci.org/tiny-dnn/tiny-dnn) [![License](https://img.shields.io/badge/license-GNU--v3.0-blue.svg)](https://raw.githubusercontent.com/tiny-dnn/tiny-dnn/master/LICENSE)

**deepworks** is a C++ framework for deep learning. We integrate acceleration libraries such as [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) to maximize performance.


## Table of contents

* [Features](#features)
* [Documentation](#documentation)
* [Dependencies](#dependencies)
* [Build from source](#build-from-source)
* [Examples](#examples)
* [Samples](#samples)
* [Comparison with other libraries](#comparison-with-other-libraries)
* [License](#license)


## Features
### Layers
  - [ReLU](./documentation/layers.md#ReLU)
  - [Leaky ReLU](./documentation/layers.md#Leaky-ReLU)
  - [ELU](./documentation/layers.md#ELU)
  - [Sigmoid](./documentation/layers.md#Sigmoid)
  - [SoftMax](./documentation/layers.md#SoftMax)
  - [Convolution](./documentation/layers.md#Convolution)
  - [MaxPooling](./documentation/layers.md#MaxPooling)
  - [GlobalAvgPooling](./documentation/layers.md#GlobalAvgPooling)
  - [Linear](./documentation/layers.md#Linear)
  - [BatchNormalization1D](./documentation/layers.md#BatchNormalization1D)
  - [BatchNormalization2D](./documentation/layers.md#BatchNormalization2D)
  - [Dropout](./documentation/layers.md#Dropout)

### Loss functions
* `CrossEntropyLoss` - criterion combines Log and NLLLoss. The input is expected to contain normalized scores (after SoftMax) for each class.

### Metrics
* `accuracy` - compute the frequency with which predictions matches labels.
* `accuracyOneHot` - compute the frequency with which predictions matches one-hot labels.
### Initializers
* `zeros` - fills the tensor with zeros.
* `constant` -  fills the tensor with `value`.
* `xavierUniform` - fills the input Tensor with values according to the method described in Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.
* `uniform` - fills the tensor with values drawn from the uniform distribution U(lower, upper).

### Optimization algorithms
* [Stochastic gradient descent](./include/deepworks/optimizers.md#Stochastic-gradient-descent)
* [Stochastic gradient descent with momentum](./include/deepworks/optimizers.md#Stochastic-gradient-descent-with-momentum)
* [Adam](./include/deepworks/optimizers.md#Adam)

### Serialization

`save_state` - save model weights in .bin file.
`load_state` - load model weights from .bin file to model.

`save_cfg` - save model architecture to .bin file.
`load_cfg` - load model architecture from .bin file.

`save` - save model weights and config to .bin files.
`load` - load model weights and config from .bin files.

`save_dot` - dump model architecture to .dot file for vizualization.

### Supported data formats
* PNG/JPEG images
* CSV

## Documentation
* [DataLoader](./documentation/data.md#DataLoader)
* [IDataset](./documentation/data.md#IDataset)
* [Model](./documentation/model.md)
* [Parameter](./documentation/parameter.md)
* [Placeholder](./documentation/placeholder.md)
* [Tensor](./documentation/tensor.md)

## Dependencies
Install dependencies for image reader

On Linux
```
sudo apt install libpng-dev
sudo apt install libjpeg-dev
```
On MacOS
```
brew install libpng
brew install jpeg
```
## Build from source

```
git clone https://github.com/NeuralDeepWorks/deepworks.git
git submodule init
git submodule update --recursive
git lfs pull
```
```
cmake ..
make -j8
```


Some cmake options are available:

|options|description|default|additional requirements to use|
|-----|-----|----|----|
|BUILD_TESTS|Build unit tests|ON<sup>1</sup>|-|
|WITH_EIGEN|Build prolect with Eigen|ON<sup>2</sup>|-|
|BUILD_SAMPLES|Build samples|ON|-|
|BUILD_BENCHMARKS|Build benchmarks|ON|-|
|DOWNLOAD_DATA|Download datasets for samples/benchmarks|ON|-|


<sup>1</sup> deepworks uses [Google Test](https://github.com/google/googletest) as default framework to run unit tests. No pre-installation required, it's  automatically downloaded during CMake configuration.

<sup>2</sup> deepworks uses [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) to CPU backend. No pre-installation required, it's  automatically downloaded during CMake configuration.


## Examples
Construct simple neural network

```cpp
dw::Placeholder in(dw::Shape{64, 100});

auto out = dw::Linear(50, "linear_0")(in);
out = dw::ReLU("relu_1")(out);
out = dw::Linear(10, "linear_2")(out);
out = dw::Softmax("probs")(out);

dw::Model model(in, out);

dw::Tensor input(in.shape());
dw::initializer::uniform(input);

model.compile();

dw::Tensor output(model.outputs()[0].shape());
model.forward(input, output);
```
## Samples
* [IRIS sample](./samples/iris/README.md)
* [MNIST sample](./samples/mnist/README.md)
* [CIFAR10 sample](./samples/cifar10/README.md)
* [CIFAR100 sample](./samples/cifar100/README.md)

## Comparison with other libraries
* [MNIST](./benchmarks/mnist/README.md)
* [CIFAR-10](./benchmarks/cifar10/README.md)

## License
GNU General Public License v3.0

