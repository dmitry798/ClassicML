cmake_minimum_required(VERSION 3.15)
project(classicml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
  add_compile_options(/W0)
endif()

# Найти Python и dev-заголовки (Python.h)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11 и его CMake-хелперы
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Ваша статическая библиотека
add_library(classicml_lib STATIC
  src/matrix.cpp
  src/optimization.cpp
  src/preprocessor.cpp
  src/models.cpp
  src/errors.cpp
  src/models/LogisticRegression.cpp
  src/models/KnnClassifier.cpp
  src/models/KnnRegression.cpp
  src/models/LinearRegression.cpp
)

target_include_directories(classicml_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# (Опционально) Принудительное подключение expr.h только в реализации
if(MSVC)
  target_compile_options(classicml_lib PRIVATE /FI"ClassicML/expr.h")
else()
  target_compile_options(classicml_lib PRIVATE -include ClassicML/expr.h)
endif()

# Python extension-модуль
pybind11_add_module(classicml bindings.cpp)
target_compile_definitions(classicml PRIVATE CLASSICML_PYBINDINGS)

# Слинковать с вашей библиотекой и Python
target_link_libraries(classicml PRIVATE classicml_lib Python::Python)

if(WIN32)
  set_target_properties(classicml PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
    OUTPUT_NAME "classicml"
  )
endif()