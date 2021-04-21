set (BENCHMARK_NAME benchmark_mnist)

add_executable(${BENCHMARK_NAME} main.cpp)

find_package(Torch 1.8.1 REQUIRED)

target_link_libraries(${BENCHMARK_NAME} ${PROJECT_NAME})
target_link_libraries(${BENCHMARK_NAME} ${TORCH_LIBRARIES})
