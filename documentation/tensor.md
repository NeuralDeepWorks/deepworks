# Tensor
Constructs a tensor with `data`.

## Methods
* `data` - returns a pointer to tensor data.
* `total` - returns number of elements in tensor.
* `empty` - checks tensor is empty.
* `strides` - returns strides of tensor data.
* `shape` - returns shape of tensor.
* `copyTo` - copies the tensor to another one.
* `allocate` - allocate tensor according to specified shape.
* `reshape` - returns a tensor with the same data and number of elements as input, but with the specified shape.
* `zeros` - fills the tensor with zeros.
* `constant` -  fills the tensor with `value`.
* `xavierUniform` - fills the input Tensor with values according to the method described in Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.
* `uniform` - fills the tensor with values drawn from the uniform distribution U(lower, upper).

#### Example
```cpp
dw::Tensor tensor(const Shape& shape, float* data);
```
