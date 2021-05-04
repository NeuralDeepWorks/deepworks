# Training simple network on CIFAR10 dataset
## Download CIFAR10 dataset
```
cd /tmp
wget -O CIFAR10.zip https://github.com/SemicolonStruggles/CIFAR-10-JPG/archive/refs/heads/master.zip
unzip CIFAR10.zip
mv CIFAR-10-JPG-master CIFAR10
```

## Build & Run
```
cd <deepworks-build>
cmake ../ -DBUILD_SAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j8
./bin/sample_cifar10_train /tmp/CIFAR10 <batch_size> <num_epochs> <dump-frequency>
```
