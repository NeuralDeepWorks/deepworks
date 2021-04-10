include(FetchContent)

find_package(Torch 1.8.1 QUIET
             PATHS ${PROJECT_SOURCE_DIR}/thirdparty/libtorch
             NO_CMAKE_PATH)

if (NOT Torch_FOUND)
    message(STATUS "Fetching Torch")
    FetchContent_Declare(
        libtorch
        URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip"
        URL_HASH MD5=77e6ac6191bd3f71fb0229ea1c33927f
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/libtorch
    )
    FetchContent_MakeAvailable(libtorch)
    find_package(Torch 1.8.1 REQUIRED
                 PATHS ${PROJECT_SOURCE_DIR}/thirdparty/libtorch
                 NO_CMAKE_PATH)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
