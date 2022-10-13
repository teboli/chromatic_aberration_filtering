from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import numpy as np




setup(
    name='chromatic_aberration_filtering',
    version="0.1.0",
    packages=find_packages(),
    author="Thomas Eboli",
    author_email="thomas.eboli@ens-paris-saclay.fr",
    url="https://github.com/teboli/chromatic_aberration_filtering",
    description="A Python implementation of [Cheng et al., TIP'13]",
    include_dirs=np.get_include(),
    ext_modules=
        cythonize(Extension('filter_cython',
                  sources=['chromatic_aberration_filtering/filter_cython.pyx'],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp'],
                  language='c++')
                  ),
    classifiers=[
        "Programming Language :: Cython",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)

