set(SRC_FILES
    # Expression API
    ${CMAKE_CURRENT_LIST_DIR}/expression/placeholder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression/call.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression/nn.cpp

    # Model stuff
    ${CMAKE_CURRENT_LIST_DIR}/model/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model/layer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model/layer_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model/graphbuilder.cpp

    ${CMAKE_CURRENT_LIST_DIR}/metrics.cpp

    ${CMAKE_CURRENT_LIST_DIR}/initializers.cpp

    ${CMAKE_CURRENT_LIST_DIR}/loss.cpp

    # General runtime entities
    ${CMAKE_CURRENT_LIST_DIR}/runtime/tensor.cpp
    # CPU runtime
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/kernels/kernels.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/cpubackend.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/kernels/kernels.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpulayer.cpp
    # CPU layers
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpurelu.cpp

    # I/O
    ${CMAKE_CURRENT_LIST_DIR}/io/image_reader.cpp
    )

set(DeepWorks_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include/")

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

# Set public API
target_include_directories(${PROJECT_NAME}
                           PUBLIC ${DeepWorks_INCLUDE_DIR}
                           PRIVATE "${PROJECT_SOURCE_DIR}/src"
)

if (WITH_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
endif(WITH_EIGEN)

if (WITH_PNG)
    find_package(PNG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_PNG)
endif(WITH_PNG)

if (WITH_JPEG)
    find_package(JPEG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE JPEG::JPEG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_JPEG)
endif(WITH_JPEG)

target_link_libraries(${PROJECT_NAME} PRIVATE ade)
