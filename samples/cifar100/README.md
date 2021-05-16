## Training convolution network on CIFAR100 dataset

* Follow instruction to setup enviroment: [DeepWorks samples](../README.md)

* Run train:
```bash
./samples/mnist/run_cifar100_train.sh
```

Possible output:
```bash
Epoch: 22
Loss: 2.39293
Loss: 2.37879
Loss: 2.37729
Loss: 2.38294
Loss: 2.38229
Loss: 2.38789
Loss: 2.38709
Loss: 2.38916
Loss: 2.38848
Loss: 2.38968
Loss: 2.39475
Loss: 2.39076
Loss: 2.38803
Loss: 2.38514
Loss: 2.38353
Loss: 2.38105
Accuracy: 0.344551
Model saved: build/cifar100_model.bin
```

* Run test:
```bash
./samples/mnist/run_cifar100_test.sh
```

Output:
```bash
Accuracy: 0.344551
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
