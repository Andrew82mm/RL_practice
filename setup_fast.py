"""
Сборка Cython-расширения _fast для BFS-ускорения окружения Block Puzzle.

Запуск:
    python setup_fast.py build_ext --inplace

Результат: block_puzzle_env/_fast.cpython-*.so рядом с исходником.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="block_puzzle_env._fast",
    sources=["block_puzzle_env/_fast.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native"],
)

setup(
    name="block_puzzle_fast",
    ext_modules=cythonize(
        ext,
        compiler_directives={"language_level": "3"},
    ),
)
