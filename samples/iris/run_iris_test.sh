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

${DW_BUILD_PATH}/bin/sample_iris_train test ${DATASETS_DIR}/IRIS/ 64 ${DW_BUILD_PATH}/iris_model.bin
