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
# Define include directories
########################################
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)
include_directories(${COMMON_INCLUDES})

########################################
# Define output paths
########################################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

########################################
# Define basic flags
########################################
option(BUILD_TESTS "Build deepworks with tests" ON)
option(WITH_EIGEN "Build deepworks with eigen backend" ON)

add_subdirectory(thirdparty)
add_subdirectory(src)
add_subdirectory(tests)
