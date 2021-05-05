## Training simple network on MNIST dataset

* Follow instruction to setup enviroment: [DeepWorks samples](../README.md)

* Run train:
```bash
./samples/mnist/run_mnist_train.sh
```

Possible output:
```bash
Epoch: 0
Loss: 0.6001
Loss: 0.43573
Loss: 0.363727
Loss: 0.319739
Loss: 0.28952
Loss: 0.265997
Loss: 0.249167
Loss: 0.241116
Accuracy: 0.96244
Model saved: /home/atalaman/workspace/deepworks/release-build/mnist_model.bin
```

* Run test:
```bash
./samples/mnist/run_mnist_test.sh
```

Output:
```bash
Accuracy: 0.96244
```

You can also run samples using binary target directly:
* Run train:
```bash
./bin/sample_mnist_train train <path-to-deepworks>/datasets/MNIST <batch_size> <num_epochs> <dump-frequency> <path-to-dump>
```

* Run test:
```bash
./bin/sample_mnist_train test <path-to-deepworks>/datasets/MNIST <batch_size> <path-to-model>
```
