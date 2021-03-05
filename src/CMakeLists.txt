set(SRC_FILES
    ${CMAKE_CURRENT_LIST_DIR}/expression/placeholder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression/call.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression/nn.cpp

    ${CMAKE_CURRENT_LIST_DIR}/model/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model/layer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model/layer_info.cpp

    ${CMAKE_CURRENT_LIST_DIR}/runtime/tensor.cpp
    )

set(DeepWorks_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include/")

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

# Set public API
target_include_directories(${PROJECT_NAME} PUBLIC   ${DeepWorks_INCLUDE_DIR}
                                           PRIVATE "${PROJECT_SOURCE_DIR}/src/")

if (WITH_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
endif(WITH_EIGEN)

target_link_libraries(${PROJECT_NAME} PRIVATE ade)
