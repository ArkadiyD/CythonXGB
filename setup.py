
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext 
import numpy as np

extensions = [
    Extension("c_xgb_test", 
        sources = ["c_xgb_test.pyx", "c_xgb/c_xgb.cpp"], 
        language = "c++",
        extra_compile_args = ["-O3", "-std=c++11", "-Wno-unused-function"]), 
]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(
        extensions
    ),
) 
