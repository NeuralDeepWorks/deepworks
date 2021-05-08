# Model
Model groups layers into an object with training and inference features.

## Methods
* `Build` - build model from config.
* `inputs` - get model inputs.
* `outputs` - get model outputs.
* `getLayer` - get layer by name.
* `params` - get all parameters of model.
* `layers` - get model layers.
* `state` - get model state.
* `cfg` - get model config.
* `train` - set training or evaluation mode.
* `compile` - create backend instance.
* `forward` - runs forward pass to compute outputs of each layer.
* `backward` - runs backward pass to accumulate gradients of each layer.

## Example
```cpp
dw::Placeholder in(dw::Shape{32, 100});
dw::Placeholder out = dw::Linear(50, "linear_0")(in);
                out = dw::ReLU("relu_1")(out);
                out = dw::Linear(10, "linear_2")(out);
                out = dw::Softmax("probs")(out);

dw::Model model(in, out);
model.compile();

dw::Tensor input(in.shape());
dw::Tensor output(model.outputs()[0].shape());
model.forward(input, output);
```