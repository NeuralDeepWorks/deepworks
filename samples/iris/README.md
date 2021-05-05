## Training simple network on IRIS dataset

* Follow instruction to setup enviroment: [DeepWorks samples](../README.md)

* Run train:
```bash
./samples/iris/run_iris_train.sh
```

Possible output:
```bash
Epoch: 0
Loss: 1.0631
Loss: 1.01716
Loss: 1.01716
Accuracy: 0.640625
Epoch: 1
Loss: 0.895396
Loss: 0.821681
Loss: 0.821681
Accuracy: 0.640625
Epoch: 2
Loss: 0.660689
Loss: 0.631235
Loss: 0.631235
Accuracy: 0.625
Epoch: 3
Loss: 0.472015
Loss: 0.489149
Loss: 0.489149
Accuracy: 0.632812
Epoch: 4
Loss: 0.487836
Loss: 0.417334
Loss: 0.417334
Accuracy: 0.648438
```

* Run test:
```bash
./samples/iris/run_iris_test.sh
```

Output:
```bash
Accuracy: 0.648438
```

You can also run samples using binary target directly:
* Run train:
```bash
./bin/sample_iris_train train <path-to-deepworks>/datasets/IRIS <batch_size> <num_epochs> <dump-frequency> <path-to-dump>
```

* Run test:
```bash
./bin/sample_iris_train test <path-to-deepworks>/datasets/IRIS <batch_size> <path-to-model>
```
