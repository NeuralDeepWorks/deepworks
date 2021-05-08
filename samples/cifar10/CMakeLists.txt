set (SAMPLE_NAME sample_cifar10_train)

add_executable(${SAMPLE_NAME} main.cpp utility.cpp)
target_link_libraries(${SAMPLE_NAME} ${PROJECT_NAME})
