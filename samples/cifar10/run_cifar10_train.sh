#!/bin/bash

if test -z ${DW_BUILD_PATH}
then
    echo "Please specify DW_BUILD_PATH variable that is the path to deepworks binary files."
    exit
fi

if test -z ${DATASETS_DIR}
then
    echo "Please specify DATASETS_DIR variable that is the path to deepworks datasets."
    exit
fi

${DW_BUILD_PATH}/bin/sample_cifar10_train train ${DATASETS_DIR}/CIFAR10/ 128 1 1000 ${DW_BUILD_PATH}/cifar10_model.bin
