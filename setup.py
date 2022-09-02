# for single thread
# from distutils.core import setup
# from Cython.Build import cythonize
#
# ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
#
# setup(ext_modules=cythonize("filter_cython.pyx", **ext_options)
#       )


# with open MP
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "filter_cython",
        ["false_color_filtering/filter_cython.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='filter-cython-parallel',
    version="1.0.0",
    author="Thomas Eboli",
    author_email="thomas.eboli@ens-paris-saclay.fr",
    url="https://github.com/teboli/chromatic_aberration_filtering",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=cythonize(ext_modules),
)
