cmake_minimum_required(VERSION 3.15)
project(ClassicML VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(CLASSICML_SOURCES
    src/matrix.cpp
    src/preprocessor.cpp
    src/errors.cpp
    src/optimization.cpp
    src/models.cpp
    src/LinearRegression.cpp
    src/LogisticRegression.cpp
    src/KnnClassifier.cpp
    src/KnnRegression.cpp
)
# Create Python module
pybind11_add_module(_classicml bindings/bindings.cpp ${CLASSICML_SOURCES})

# Link math library on Unix
if(UNIX AND NOT APPLE)
    target_link_libraries(_classicml PRIVATE m)
endif()

# Set output directory
set_target_properties(_classicml PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/classicml
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/classicml
)

# Installation
install(TARGETS _classicml 
        LIBRARY DESTINATION classicml
        RUNTIME DESTINATION classicml)