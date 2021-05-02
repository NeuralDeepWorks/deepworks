## DeepWorks samples

Before using scripts setup enviroment:

* Build deepworks with samples:
```bash
mkdir build
cd build
cmake <path-to-deepworks-root> -DBUILD_SAMPLES=ON -DDOWNLOAD_DATA=ON
make -j8
```

```bash
export DW_BUILD_PATH=<path-to-build-folder>
export DATASETS_DIR=<path-to-deepworks-root>/datasets
```

* [MNIST sample](./mnist/README.md)
