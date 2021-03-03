set(SRC_FILES
    ${CMAKE_CURRENT_LIST_DIR}/placeholder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/call.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/layers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/layer_info.cpp
    )

set(HDR_FILES
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
   )

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
# Set public API
target_include_directories(${PROJECT_NAME}
                           PUBLIC 
                           ${HDR_FILES}
                           # FIXME: Needed only for tests,
                           # belive it will be removed.
                           "${PROJECT_SOURCE_DIR}/src")

if (WITH_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
endif(WITH_EIGEN)

# It should be PRIVATE, but now used in public header!
target_link_libraries(${PROJECT_NAME} PUBLIC ade)
