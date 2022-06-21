from pathlib import Path
from setuptools import find_packages, setup


# Package meta-data.
NAME = "ride-duration-prediction"
DESCRIPTION = ""
URL = ""
EMAIL = ""
AUTHOR = ""
REQUIRES_PYTHON = ">=3.9.0"


# The rest you shouldn"t have to touch too much. Except for install_requires=[]. 
# Perhaps also the License and Trove Classifiers if publishing to PyPI (public).
# ------------------------------------------------

long_description = DESCRIPTION

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "ride_duration"
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"ride_duration": ["VERSION"]},
    install_requires=[],
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)