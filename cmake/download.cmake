set (DATASETS_DIR "${PROJECT_SOURCE_DIR}/datasets")
if (NOT EXISTS ${DATASETS_DIR})
    file(MAKE_DIRECTORY ${DATASETS_DIR})
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