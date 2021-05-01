# Benchmarks. Comparation Deepworks and Torch frameworks.
## Download MNIST dataset
```
cd /tmp
wget -O MNIST.zip https://github.com/teavanist/MNIST-JPG/raw/master/MNIST%20Dataset%20JPG%20format.zip
unzip MNIST.zip
```

## Build & Run
```
cd <deepworks-build>
cmake ../ -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
make -j8
./bin/benchmark_mnist /tmp/MNIST <batch_size> <num_epochs>
```
