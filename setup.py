import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Helper to locate pybind11’s headers
class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        name="core_cpp",
        sources=["src/core_cpp.cpp"],
        include_dirs=[
            get_pybind_include(),            # <pybind11> headers
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
]

setup(
    name="sync_app",
    version="0.1.0",
    author="Your Name",
    description="An app that runs C++ and Python modules in sync",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
