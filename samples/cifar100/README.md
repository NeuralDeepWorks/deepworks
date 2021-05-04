# Training simple network on CIFAR100 dataset
## Download CIFAR10 dataset
```
cd /tmp
wget -O CIFAR100.zip https://github.com/SemicolonStruggles/CIFAR-100-JPG/archive/refs/heads/master.zip
unzip CIFAR100.zip
mv CIFAR-100-JPG-master CIFAR100
```

## Build & Run
```
cd <deepworks-build>
cmake ../ -DBUILD_SAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j8
./bin/sample_cifar100_train /tmp/CIFAR100 <batch_size> <num_epochs> <dump-frequency>
```
