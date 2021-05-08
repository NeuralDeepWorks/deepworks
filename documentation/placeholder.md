# Placeholder
Inserts a placeholder for a tensor that will be always fed.
Placeholder is used to build the model.
## Example
```cpp
dw::Placeholder in(dw::Shape{32, 100});
dw::Placeholder out = dw::Linear(50, "linear_0")(in);
                out = dw::ReLU("relu_1")(out);
                out = dw::Linear(10, "linear_2")(out);
                out = dw::Softmax("probs")(out);
```