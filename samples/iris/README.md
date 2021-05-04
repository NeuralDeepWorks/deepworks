# Training simple network on Iris dataset
## Download Iris dataset
```
cd /tmp
mkdir iris
cd iris
mkdir train
mkdir test
cd train
wget -O iris.zip https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv
cd ../test
wget -O iris.zip https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv
```

## Build & Run
```
cd <deepworks-build>
cmake ../ -DBUILD_SAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j8
./bin/sample_iris_train /tmp/iris <batch_size> <num_epochs> <dump-frequency>
```
