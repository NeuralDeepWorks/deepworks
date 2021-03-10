set(SRC_FILES
        ${CMAKE_CURRENT_LIST_DIR}/placeholder.cpp
        ${CMAKE_CURRENT_LIST_DIR}/call.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tensor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernels.cpp
        )

set(DeepWorks_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include/")

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

# Set public API
target_include_directories(${PROJECT_NAME}
                           PUBLIC ${DeepWorks_INCLUDE_DIR}
                           PRIVATE "${PROJECT_NAME}/src"
)

if (WITH_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
endif(WITH_EIGEN)
