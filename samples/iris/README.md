## Training simple network on IRIS dataset

* Follow instruction to setup enviroment: [DeepWorks samples](../README.md)

* Run train:
```bash
./samples/iris/run_iris_train.sh
```

Possible output:
```bash
Epoch: 0
Loss: 1.09987
Loss: 1.03461
Loss: 1.03461
Accuracy: 0.609375
Epoch: 1
Loss: 0.888799
Loss: 0.834636
Loss: 0.834636
Accuracy: 0.617188
Epoch: 2
Loss: 0.654118
Loss: 0.636887
Loss: 0.636887
Accuracy: 0.632812
Epoch: 3
Loss: 0.534014
Loss: 0.522762
Loss: 0.522762
Accuracy: 0.632812
Epoch: 4
Loss: 0.44576
Loss: 0.389611
Loss: 0.389611
Accuracy: 0.648438
Model saved: build/iris_model.bin
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
