set(TEST_NAME deepworks_unit_tests)

enable_testing()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/unit/*.cpp)

add_executable(${TEST_NAME} ${TEST_SRC_FILES})
add_definitions(-DTEST_DATA_PATH=\"${PROJECT_SOURCE_DIR}/tests/testdata\")

find_package(Torch 1.8.1 REQUIRED)

target_link_libraries(${TEST_NAME} gtest gtest_main)
target_link_libraries(${TEST_NAME} ${PROJECT_NAME})
target_link_libraries(${TEST_NAME} ${TORCH_LIBRARIES})
