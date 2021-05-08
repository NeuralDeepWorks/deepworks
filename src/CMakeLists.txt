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
    ${CMAKE_CURRENT_LIST_DIR}/model/parameter.cpp

    ${CMAKE_CURRENT_LIST_DIR}/metrics.cpp

    ${CMAKE_CURRENT_LIST_DIR}/initializers.cpp

    ${CMAKE_CURRENT_LIST_DIR}/loss.cpp

    ${CMAKE_CURRENT_LIST_DIR}/optimizer.cpp

    # General runtime entities
    ${CMAKE_CURRENT_LIST_DIR}/runtime/tensor.cpp
    # CPU runtime
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/kernels/kernels.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/cpubackend.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/kernels/kernels.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpulayer.cpp
    # CPU layers
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpurelu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpulinear.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpusoftmax.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpubatchnorm1d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpubatchnorm2d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpuelu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpuleakyrelu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpuconvolution.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpumaxpooling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpusigmoid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpudropout.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpuglobalavgpooling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime/cpu/layers/cpuadd.cpp

    # I/O
    ${CMAKE_CURRENT_LIST_DIR}/io/image_reader.cpp

    # Utils
    ${CMAKE_CURRENT_LIST_DIR}/utils/assert.cpp

    # DataLoader
    ${CMAKE_CURRENT_LIST_DIR}/dataloader.cpp

    # Serialization
    ${CMAKE_CURRENT_LIST_DIR}/serialization.cpp
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

find_package(PNG)
if (PNG_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_PNG)
endif(PNG_FOUND)

find_package(JPEG)
if (JPEG_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE JPEG::JPEG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_JPEG)
endif(JPEG_FOUND)

target_link_libraries(${PROJECT_NAME} PRIVATE ade)

# NB: To allow Eigen use multiple threads.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
