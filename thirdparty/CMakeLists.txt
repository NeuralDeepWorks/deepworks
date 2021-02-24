if (BUILD_TESTS)
    add_subdirectory(googletest)
endif(BUILD_TESTS)

if (WITH_EIGEN)
    add_subdirectory(eigen)
endif(WITH_EIGEN)
