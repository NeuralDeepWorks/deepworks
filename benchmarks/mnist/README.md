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

### Possible output:

```****************************************************************************************
*                                     Dataset info                                     *
****************************************************************************************
- Name: MNIST
- Train size: 60000
- Validation size: 10000
- Image shape: 28x28
****************************************************************************************
*                                        Torch                                        *
****************************************************************************************
- Epoch:          1/10 | Train loss: 0.429816   | Validation accuracy: 0.9361    
- Epoch:          2/10 | Train loss: 0.221335   | Validation accuracy: 0.9522    
- Epoch:          3/10 | Train loss: 0.170636   | Validation accuracy: 0.959     
- Epoch:          4/10 | Train loss: 0.141698   | Validation accuracy: 0.9656    
- Epoch:          5/10 | Train loss: 0.120021   | Validation accuracy: 0.9676    
- Epoch:          6/10 | Train loss: 0.104744   | Validation accuracy: 0.972     
- Epoch:          7/10 | Train loss: 0.0938143  | Validation accuracy: 0.973     
- Epoch:          8/10 | Train loss: 0.0849775  | Validation accuracy: 0.9736    
- Epoch:          9/10 | Train loss: 0.0770597  | Validation accuracy: 0.9718    
- Epoch:         10/10 | Train loss: 0.0706118  | Validation accuracy: 0.9749    
****************************************************************************************
*                                      Deepworks                                      *
****************************************************************************************
- Epoch:          1/10 | Train loss: 0.460949   | Validation accuracy: 0.936599  
- Epoch:          2/10 | Train loss: 0.217663   | Validation accuracy: 0.951923  
- Epoch:          3/10 | Train loss: 0.160586   | Validation accuracy: 0.960737  
- Epoch:          4/10 | Train loss: 0.129075   | Validation accuracy: 0.966046  
- Epoch:          5/10 | Train loss: 0.106719   | Validation accuracy: 0.96865   
- Epoch:          6/10 | Train loss: 0.09366    | Validation accuracy: 0.96845   
- Epoch:          7/10 | Train loss: 0.0829493  | Validation accuracy: 0.970152  
- Epoch:          8/10 | Train loss: 0.0730025  | Validation accuracy: 0.973057  
- Epoch:          9/10 | Train loss: 0.0673074  | Validation accuracy: 0.971354  
- Epoch:         10/10 | Train loss: 0.0588462  | Validation accuracy: 0.974559  
========================================================================================
||                    ||     Deepworks      ||       Torch        ||    DW vs Torch     ||
========================================================================================
||       Epochs       ||         10         ||         10         ||        None        ||
||     Train time     ||      32552 ms      ||      26322 ms      ||      -23.67%       ||
||   Inference time   ||      2034 ms       ||      2799 ms       ||      +37.61%       ||
||      Accuracy      ||      0.974559      ||      0.974900      ||       -0.03%       ||
||        Loss        ||      0.058846      ||      0.070612      ||      +19.99%       ||
========================================================================================```
