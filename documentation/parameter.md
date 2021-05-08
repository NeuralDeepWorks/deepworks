# Parameter
Parameter contains tensors with data and gradients. Used to store trainable parameters of `Model`.
## Methods
* `data` - returns tensor with initial data.
* `grad` - if `is_trainable == true` then attribute will contain the gradients computed and future calls to `backward()` will accumulate (add) gradients into it, otherwise grad is an empty tensor.
* `train` - set `is_trainable` to `true`.
* `is_trainable` - is `true` if gradients need to be computed for this `Tensor`, false otherwise.
