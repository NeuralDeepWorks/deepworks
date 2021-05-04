set (SAMPLE_NAME sample_cifar100_train)

add_executable(${SAMPLE_NAME} main.cpp utility.cpp)
target_link_libraries(${SAMPLE_NAME} ${PROJECT_NAME})
