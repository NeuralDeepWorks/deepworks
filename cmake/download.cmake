set (DATASETS_DIR "${PROJECT_SOURCE_DIR}/datasets")
if (NOT EXISTS ${DATASETS_DIR})
    file(MAKE_DIRECTORY ${DATASETS_DIR})
endif()

# NB: Download Iris dataset if it doesn't exist
if (NOT EXISTS "${DATASETS_DIR}/IRIS")
    message(STATUS "Downloading IRIS dataset")
    file(MAKE_DIRECTORY "${DATASETS_DIR}/IRIS/train")
    file(MAKE_DIRECTORY "${DATASETS_DIR}/IRIS/test")

    file(DOWNLOAD
         https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv
         "${DATASETS_DIR}/IRIS/iris.csv" SHOW_PROGRESS)

     file(COPY "${DATASETS_DIR}/IRIS/iris.csv" DESTINATION "${DATASETS_DIR}/IRIS/train/")
     file(COPY "${DATASETS_DIR}/IRIS/iris.csv" DESTINATION "${DATASETS_DIR}/IRIS/test/")
     file(REMOVE "${DATASETS_DIR}/IRIS/iris.csv")
endif()

# NB: Download MNIST dataset if it doesn't exist
if (NOT EXISTS "${DATASETS_DIR}/MNIST")
    message(STATUS "Downloading MNIST dataset")
    file(DOWNLOAD
         https://github.com/teavanist/MNIST-JPG/raw/master/MNIST%20Dataset%20JPG%20format.zip
         "${DATASETS_DIR}/MNIST.zip" SHOW_PROGRESS)

    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "MNIST.zip"
                    WORKING_DIRECTORY "${DATASETS_DIR}")

    file(RENAME "${DATASETS_DIR}/MNIST Dataset JPG format"     "${DATASETS_DIR}/MNIST")
    file(RENAME "${DATASETS_DIR}/MNIST/MNIST - JPG - testing"  "${DATASETS_DIR}/MNIST/test")
    file(RENAME "${DATASETS_DIR}/MNIST/MNIST - JPG - training" "${DATASETS_DIR}/MNIST/train")
    file(REMOVE "${DATASETS_DIR}/MNIST.zip")
endif()

# NB: Download CIFAR10 dataset if it doesn't exist
if (NOT EXISTS "${DATASETS_DIR}/CIFAR10")
    message(STATUS "Downloading CIFAR10 dataset")
    file(DOWNLOAD
         https://github.com/SemicolonStruggles/CIFAR-10-JPG/archive/refs/heads/master.zip
         "${DATASETS_DIR}/CIFAR10.zip" SHOW_PROGRESS)

    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "CIFAR10.zip"
                    WORKING_DIRECTORY "${DATASETS_DIR}")

    file(RENAME "${DATASETS_DIR}/CIFAR-10-JPG-master" "${DATASETS_DIR}/CIFAR10")
    file(REMOVE "${DATASETS_DIR}/CIFAR10.zip")
endif()

# NB: Download CIFAR100 dataset if it doesn't exist
if (NOT EXISTS "${DATASETS_DIR}/CIFAR100")
    message(STATUS "Downloading CIFAR100 dataset")
    file(DOWNLOAD
         https://github.com/SemicolonStruggles/CIFAR-100-JPG/archive/refs/heads/master.zip
         "${DATASETS_DIR}/CIFAR100.zip" SHOW_PROGRESS)

    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "CIFAR100.zip"
                    WORKING_DIRECTORY "${DATASETS_DIR}")

    file(RENAME "${DATASETS_DIR}/CIFAR-100-JPG-master" "${DATASETS_DIR}/CIFAR100")
    file(REMOVE "${DATASETS_DIR}/CIFAR100.zip")
endif()
