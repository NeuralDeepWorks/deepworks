# Optimization algorithms
### Stochastic gradient descent
Implements stochastic gradient descent.
#### Attributes
- `lr` – step size
#### Example
```cpp
float lr = 1e-3f;
dw::optimizer::SGD opt(model.params(), lr);

...

loss = criterion.forward(predict, y);
criterion.backward(predict, y, grad_output);
model.backward(X, predict, grad_output);

opt.step();
```

### Stochastic gradient descent with momentum
Implements stochastic gradient descent with momentum. It is a technique for accelerating gradient descent that accumulates a velocity
vector in directions of persistent reduction in the objective across iterations.
#### Attributes
- `lr` – step size
- `gamma` – exponential decay rate for the velocity
vector (default: 0.9)
#### Example
```cpp
float lr = 1e-3f;
float gamma = 0.95f;
dw::optimizer::SGDMomentum opt(model.params(), lr, gamma);
// dw::optimizer::SGDMomentum opt(model.params(), lr);

...

loss = criterion.forward(predict, y);
criterion.backward(predict, y, grad_output);
model.backward(X, predict, grad_output);

opt.step();
```

### Adam
Implements Adam algorithm. It computes the decaying averages of past and past squared gradients, which are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively. Bias-corrected first and second moment estimates are used to update the parameters.
#### Attributes
- `lr` – step size
- `beta1` – exponential decay rate for the first moment estimates (default: 0.9)
- `beta2` – exponential decay rate for the second moment estimates (default: 0.999)
- `epsilon` – small number to prevent any division by zero in the implementation (default: 1e-3)
#### Example
```cpp
float lr = 1e-3f;
float epsilon = 1e-5f;
std::array<float, 2> betas = {0.9, 0.999};
dw::optimizer::Adam opt(model.params, lr, betas, epsilon);
// dw::optimizer::Adam opt(model.params, lr);

...

loss = criterion.forward(predict, y);
criterion.backward(predict, y, grad_output);
model.backward(X, predict, grad_output);

opt.step();
```
