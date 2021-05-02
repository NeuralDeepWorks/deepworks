# deepworks

[![Build Status](https://travis-ci.org/NeuralDeepWorks/deepworks.svg?branch=main)](https://travis-ci.org/tiny-dnn/tiny-dnn) [![License](https://img.shields.io/badge/license-GNU--v3.0-blue.svg)](https://raw.githubusercontent.com/tiny-dnn/tiny-dnn/master/LICENSE)

**deepworks** is a C++ framework for deep learning. We integrate acceleration libraries such as [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) to maximize performance.


## Table of contents

* [Features](#features)
* [Dependencies](#dependencies)
* [Build from source](#build-from-source)
* [Examples](#examples)
* [License](#license)
* [Comparison with other libraries](#comparison-with-other-libraries)


## Features
### layers
  - [ReLU](./include/deepworks/layers.md#ReLU)
  - [Leaky ReLU](./include/deepworks/layers.md#Leaky-ReLU)
  - [ELU](./include/deepworks/layers.md#ELU)
  - [Sigmoid](./include/deepworks/layers.md#Sigmoid)
  - [SoftMax](./include/deepworks/layers.md#SoftMax)
  - [Convolution](./include/deepworks/layers.md#Convolution)
  - [MaxPooling](./include/deepworks/layers.md#MaxPooling)
  - [GlobalAvgPooling](./include/deepworks/layers.md#GlobalAvgPooling)
  - [Linear](./include/deepworks/layers.md#Linear)
  - [BatchNormalization1D](./include/deepworks/layers.md#BatchNormalization1D)
  - [BatchNormalization2D](./include/deepworks/layers.md#BatchNormalization2D)
  - [Dropout](./include/deepworks/layers.md#Dropout)

### loss functions
* cross-entropy

### optimization algorithms
* stochastic gradient descent (with/without L2 normalization)
* stochastic gradient descent with momentum
* adam

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

## License
GNU General Public License v3.0

## Comparison with other libraries
TODO
