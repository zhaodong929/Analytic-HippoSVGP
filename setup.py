from setuptools import setup, find_packages

setup(
    name="HIPPO-SVGP",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "pytest",
        "tqdm",
        "numpy",
        "pandas",
        "torchvision",
        "torch",
        "ipywidgets",
        "jupyter",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "cdsapi>=0.7.4",
        "xarray",
    ],
)