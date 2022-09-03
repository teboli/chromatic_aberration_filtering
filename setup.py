from setuptools import Extension, setup

# By default use the .c I generated. If needs be,
# set the flag to True to import and compile with
# Cython on your own computer.
USE_CYTHON =  True

ext = '.pyx' if USE_CYTHON else '.c'

ext_modules = [
    Extension(
        "chromatic_aberration_filtering.filter_cython",
        ["chromatic_aberration_filtering/filter_cython" + ext],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

if USE_CYTHON:
    from Cython.Build import cythonize
    ext_modules=cythonize(ext_modules)

setup(
    name='chromatic_aberration_filtering',
    version="1.0.0",
    author="Thomas Eboli",
    author_email="thomas.eboli@ens-paris-saclay.fr",
    url="https://github.com/teboli/chromatic_aberration_filtering",
    packages=['chromatic_aberration_filtering'],
    package_data = {
        'chromatic_aberration_filtering': ['*.pxd']
        },
    include_package_data=True,
    ext_modules=ext_modules,
    zip_safe=False,
)
