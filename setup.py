from setuptools import setup, Extension, find_packages

core_ext = Extension(
    name="tropiq._core",
    sources=["csrc/maxplus_matvec.c", "csrc/python_bindings.c"],
)

setup(
    name="tropiq",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=[core_ext],
)
