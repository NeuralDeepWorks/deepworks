# Layers
### ReLU
Applies the rectified linear unit function element-wise.
ReLU(x) = max(0, x)
#### Example
```cpp
dw::Placeholder in;
auto out = dw::ReLU("relu")(in);
```

### Leaky ReLU
Applies the element-wise function:
LeakyRELU(x) = x if x≥0 else alpha*x
#### Attributes
* `alpha` – controls the angle of the negative slope

#### Example
```cpp
float alpha = 0.001;
dw::Placeholder in;
auto out = dw::LeakyReLU(alpha, "leaky_relu")(in);
```

### ELU
Applies the element-wise function:
ELU(x) = x if x≥0 else alpha*(exp(x) - 1)
#### Attributes
* `alpha` – scale for the negative factor

#### Example
```cpp
float alpha = 1.f;
dw::Placeholder in;
auto out = dw::ELU(alpha, "elu")(in);
```

### Sigmoid
Applies the element-wise function:
Sigmoid(x) = 1 / (1 + exp(-x))

#### Example
```cpp
dw::Placeholder in;
auto out = dw::Sigmoid("sigmoid")(in);
```

### SoftMax
Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1. Softmax will be computed along `axis=1` (so every slice along `axis=1` will sum to 1).
#### Example
```cpp
dw::Placeholder in;
auto out = dw::SoftMax("softmax")(in);
```

### Convolution
Applies a 2D convolution over an input image composed of several input planes.
#### Attributes
* `out_channels` – number of channels produced by the convolution
* `kernel` - the size of the sliding window
* `strides` - controls the stride for the cross-correlation.
* `padding` - controls the amount of implicit padding on both sides for padding number of points for each dimension.

#### Example
```cpp
int out_channels = 4;
std::array<int, 2> kernel_conv{5, 5};
std::array<int, 2> padding_conv{2, 2};
std::array<int, 2> stride_conv{1, 1};
dw::Placeholder in;
auto out = dw::Convolution(out_channels, kernel_conv, padding_conv, stride_conv, "conv")(in);
```

### MaxPooling
Applies a 2D max pooling over an input image composed of several input planes.
#### Attributes
* `kernel` - the size of the sliding window
* `strides` - controls the stride for the max pooling.
* `padding` - controls the amount of implicit padding on both sides for padding number of points for each dimension.

#### Example
```cpp
std::array<int, 2> kernel_pool{2, 2};
std::array<int, 2> padding_pool{0, 0};
std::array<int, 2> stride_pool{2, 2};
dw::Placeholder in;
auto out = dw::MaxPooling(kernel_pool, padding_pool, stride_pool, "pool")(in);
```

### GlobalAvgPooling
The GlobalAvgPooling block takes a tensor of size (input channels) x (input height) x (input width) and computes the average value of all values across the entire (input height) x (input width) matrix for each of the (input channels).
#### Example
```cpp
dw::Placeholder in;
auto out = dw::GlobalAvgPooling("global_avg_pool")(in);
```

### Linear
Applies a linear transformation to the incoming data: $$y = xA^T + b$$
#### Attributes
* `units` – size of each output sample
#### Example
```cpp
int units = 10;
dw::Placeholder in;
auto out = dw::Linear(units, "linear")(in);
```

### BatchNormalization1D
Applies Batch Normalization over a 2D input.
The mean and standard-deviation are calculated per-dimension over the mini-batches, `gamma` and `beta` are learnable parameter vectors of size C. By default, the elements of `gamma` are set to 1 and the elements of `beta` are set to 0. Also by default, during training this layer keeps running estimates of its computed mean and variance, which are then used for normalization during evaluation.
#### Attributes
* `eps` -  a value added to the denominator for numerical stability.
* `alpha` - the momentum used for the `running_mean` and `running_var` computation.

#### Example
```cpp
float eps = 1e-5f;
float alpha = 0.1;
dw::Placeholder in;
auto out = dw::BatchNorm1D(epsilon, alpha, "batchnorm1d")(in);
```

### BatchNormalization2D
Applies Batch Normalization over a 4D input.
The mean and standard-deviation are calculated per-dimension over the mini-batches, `gamma` and `beta` are learnable parameter vectors of size C. By default, the elements of `gamma` are set to 1 and the elements of `beta` are set to 0. Also by default, during training this layer keeps running estimates of its computed mean and variance, which are then used for normalization during evaluation.
#### Attributes
* `eps` -  a value added to the denominator for numerical stability.
* `alpha` - the momentum used for the `running_mean` and `running_var` computation.

#### Example
```cpp
float eps = 1e-5f;
float alpha = 0.1;
dw::Placeholder in;
auto out = dw::BatchNorm2D(epsilon, alpha, "batchnorm2d")(in);
```
### Dropout
During training, mask with uniform distribution is generated. And elements with probability < p are nullified.

This has proven to be a technique for regularization and preventing the co-adaptation of neurons.
The outputs are scaled by a factor of 1 / (1 - p) during training. This means that during evaluation the module simply computes an identity function.
#### Attributes
* `p` – probability of an element to be zeroed
#### Example
```cpp
float p = 0.2f;
dw::Placeholder in;
auto out = dw::Dropout(p, "dropout")(in);
```
