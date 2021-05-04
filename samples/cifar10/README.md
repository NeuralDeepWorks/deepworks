## Training convolution network on CIFAR10 dataset

* Follow instruction to setup enviroment: [DeepWorks samples](../README.md)

* Run train:
```bash
./samples/mnist/run_cifar10_train.sh
```

Possible output:
```bash
Epoch: 0
Loss: 2.21671
Loss: 2.04742
Loss: 1.92269
Loss: 1.91002
Accuracy: 0.343249
Model saved: build/cifar10_model.bin
```

* Run test:
```bash
./samples/mnist/run_cifar10_test.sh
```

Output:
```bash
Accuracy: 0.343249
```

You can also run samples using binary target directly:
* Run train:
```bash
./bin/sample_cifar10_train train <path-to-deepworks>/datasets/CIFAR10 <batch_size> <num_epochs> <dump-frequency>
```

* Run test:
```bash
./bin/sample_cifar10_train test <path-to-deepworks>/datasets/CIFAR10 <batch_size>
```
