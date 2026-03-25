cmake_minimum_required(VERSION 3.18)

project(OmniGBDT VERSION 0.0.1 LANGUAGES CXX)

option(OMNIGBDT_BUILD_EXAMPLE "Build the native example executable" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OMNIGBDT_CORE_SOURCES
    src/booster.cpp
    src/dataStruct.cpp
    src/io.cpp
    src/loss.cpp
    src/mathFunc.cpp
    src/tree.cpp
)

set(OMNIGBDT_LIBRARY_SOURCES
    src/api.cpp
    ${OMNIGBDT_CORE_SOURCES}
)

add_library(omnigbdt_native SHARED ${OMNIGBDT_LIBRARY_SOURCES})
target_include_directories(omnigbdt_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(omnigbdt_native PRIVATE OMNIGBDT_BUILDING_LIBRARY=1)
set_target_properties(
    omnigbdt_native
    PROPERTIES
        OUTPUT_NAME "_omnigbdt"
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

find_package(OpenMP COMPONENTS CXX QUIET)

if(OpenMP_CXX_FOUND)
    target_link_libraries(omnigbdt_native PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    find_path(
        OMNIGBDT_OPENMP_INCLUDE_DIR
        omp.h
        PATHS
            /opt/homebrew/opt/libomp/include
            /usr/local/opt/libomp/include
    )
    find_library(
        OMNIGBDT_OPENMP_LIBRARY
        NAMES omp libomp
        PATHS
            /opt/homebrew/opt/libomp/lib
            /usr/local/opt/libomp/lib
    )

    if(OMNIGBDT_OPENMP_INCLUDE_DIR AND OMNIGBDT_OPENMP_LIBRARY)
        target_compile_options(omnigbdt_native PRIVATE -Xpreprocessor -fopenmp)
        target_include_directories(omnigbdt_native PUBLIC ${OMNIGBDT_OPENMP_INCLUDE_DIR})
        target_link_libraries(omnigbdt_native PRIVATE ${OMNIGBDT_OPENMP_LIBRARY})
    else()
        message(FATAL_ERROR "OpenMP was not found. Install libomp or set OpenMP_ROOT.")
    endif()
else()
    message(FATAL_ERROR "OpenMP was not found. Install an OpenMP-enabled toolchain or set OpenMP_ROOT.")
endif()

install(
    TARGETS omnigbdt_native
    LIBRARY DESTINATION "omnigbdt"
    RUNTIME DESTINATION "omnigbdt"
)

if(OMNIGBDT_BUILD_EXAMPLE)
    add_executable(omnigbdt_example src/main.cpp ${OMNIGBDT_CORE_SOURCES})
    target_include_directories(omnigbdt_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(omnigbdt_example PRIVATE OpenMP::OpenMP_CXX)
    elseif(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        target_compile_options(omnigbdt_example PRIVATE -Xpreprocessor -fopenmp)
        target_include_directories(omnigbdt_example PRIVATE ${OMNIGBDT_OPENMP_INCLUDE_DIR})
        target_link_libraries(omnigbdt_example PRIVATE ${OMNIGBDT_OPENMP_LIBRARY})
    endif()
endif()
