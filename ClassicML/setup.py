from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os

# Все исходные файлы
sources = [
    "bindings.cpp",
    "src/matrix.cpp",
    "src/preprocessor.cpp",
    "src/errors.cpp",
    "src/optimization.cpp",
    "src/models.cpp",
    "src/models/LinearRegression.cpp",
    "src/models/LogisticRegression.cpp",
    "src/models/KnnClassifier.cpp",
    "src/models/KnnRegression.cpp",
    "src/models/KMeans.cpp",
]

# Дополнительные флаги для компилятора
extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    # Windows (MSVC)
    extra_compile_args = [
        '/std:c++14',
        '/bigobj',           # Для больших объектных файлов
        '/EHsc',             # Exception handling
        '/O2',               # Optimization
    ]
else:
    # Linux/Mac (GCC/Clang)
    extra_compile_args = [
        '-std=c++14',
        '-O3',
        '-Wall',
    ]

# Создание расширения
ext_modules = [
    Pybind11Extension(
        "classicml._classicml",
        sources,
        include_dirs=["include"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=14,
    ),
]

setup(
    name="ClassicML",
    version="1.0.0",
    author="Your Name",
    description="Machine Learning library from scratch in C++",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["classicml"],
    python_requires=">=3.7",
    install_requires=["numpy>=1.19.0"],
    zip_safe=False,
)
