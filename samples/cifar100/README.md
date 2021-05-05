## Training convolution network on CIFAR100 dataset

* Follow instruction to setup enviroment: [DeepWorks samples](../README.md)

* Run train:
```bash
./samples/mnist/run_cifar100_train.sh
```

Possible output:
```bash
Epoch: 0
Loss: 4.60511
Loss: 4.55751
Loss: 4.44311
Loss: 4.42878
Accuracy: 0.0556891
Model saved: build/cifar100_model.bin
```

* Run test:
```bash
./samples/mnist/run_cifar100_test.sh
```

Output:
```bash
Accuracy: 0.0556891
```

You can also run samples using binary target directly:
* Run train:
```bash
./bin/sample_cifar100_train train <path-to-deepworks>/datasets/CIFAR100 <batch_size> <num_epochs> <dump-frequency> <path-to-dump>
```

* Run test:
```bash
./bin/sample_cifar100_train test <path-to-deepworks>/datasets/CIFAR100 <batch_size> <path-to-model>
```
