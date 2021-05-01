cmake_minimum_required(VERSION 3.12)

########################################
# Project setup
########################################
project(deepworks)

########################################
# Set up the compiler flags
########################################
set(CMAKE_CXX_STANDARD 17)

########################################
# Define output paths
########################################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

########################################
# Define basic flags
########################################
option(BUILD_TESTS   "Build deepworks with tests"         ON)
option(WITH_EIGEN    "Build deepworks with eigen backend" ON)
option(BUILD_SAMPLES "Build samples"                      ON)
option(DOWNLOAD_DATA "Download and unpack datasets"       ON)

if (DOWNLOAD_DATA)
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
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(thirdparty)
add_subdirectory(src)
add_subdirectory(tests)

if (BUILD_SAMPLES)
    add_subdirectory(samples)
endif()
