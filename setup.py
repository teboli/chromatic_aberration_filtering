# for single thread
# from distutils.core import setup
# from Cython.Build import cythonize
#
# ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
#
# setup(ext_modules=cythonize("filter_cython.pyx", **ext_options)
#       )


# with open MP
from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "filter_cython",
        ["filter_cython.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='filter-cython-parallel',
    ext_modules=cythonize(ext_modules),
)
