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

### Possible output:

```****************************************************************************************
*                                     Dataset info                                     *
****************************************************************************************
- Name: CIFAR10
- Train size: 50000
- Validation size: 10000
- Element shape: [3, 32, 32]
****************************************************************************************
*                                        Torch                                        *
****************************************************************************************
- Epoch:          1/10 | Train loss: 1.6499     | Validation accuracy: 0.3757    
- Epoch:          2/10 | Train loss: 1.39314    | Validation accuracy: 0.366     
- Epoch:          3/10 | Train loss: 1.2735     | Validation accuracy: 0.4418    
- Epoch:          4/10 | Train loss: 1.20441    | Validation accuracy: 0.3869    
- Epoch:          5/10 | Train loss: 1.14712    | Validation accuracy: 0.3817    
- Epoch:          6/10 | Train loss: 1.09779    | Validation accuracy: 0.413     
- Epoch:          7/10 | Train loss: 1.04985    | Validation accuracy: 0.4447    
- Epoch:          8/10 | Train loss: 1.01513    | Validation accuracy: 0.4811    
- Epoch:          9/10 | Train loss: 0.972145   | Validation accuracy: 0.4957    
- Epoch:         10/10 | Train loss: 0.937714   | Validation accuracy: 0.5577    
****************************************************************************************
*                                      Deepworks                                      *
****************************************************************************************
- Epoch:          1/10 | Train loss: 1.49793    | Validation accuracy: 0.5628    
- Epoch:          2/10 | Train loss: 1.08175    | Validation accuracy: 0.619691  
- Epoch:          3/10 | Train loss: 0.918944   | Validation accuracy: 0.642428  
- Epoch:          4/10 | Train loss: 0.809271   | Validation accuracy: 0.681891  
- Epoch:          5/10 | Train loss: 0.748476   | Validation accuracy: 0.66276   
- Epoch:          6/10 | Train loss: 0.665355   | Validation accuracy: 0.688702  
- Epoch:          7/10 | Train loss: 0.607601   | Validation accuracy: 0.689203  
- Epoch:          8/10 | Train loss: 0.538813   | Validation accuracy: 0.698217  
- Epoch:          9/10 | Train loss: 0.484722   | Validation accuracy: 0.695012  
- Epoch:         10/10 | Train loss: 0.431087   | Validation accuracy: 0.699119  

========================================================================================
||                    ||     Deepworks      ||       Torch        ||    DW vs Torch     ||
========================================================================================
||       Epochs       ||         10         ||         10         ||        None        ||
||     Train time     ||     1973741 ms     ||     359357 ms      ||      -449.24%      ||
||   Inference time   ||      91478 ms      ||      26570 ms      ||      -244.29%      ||
||      Accuracy      ||      0.699119      ||      0.557700      ||      +25.36%       ||
||        Loss        ||      0.431087      ||      0.937714      ||      +117.52%      ||
========================================================================================```
