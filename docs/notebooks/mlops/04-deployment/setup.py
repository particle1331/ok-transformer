from pathlib import Path
from setuptools import find_packages, setup


# Package meta-data.
NAME = 'ride-duration-prediction'
DESCRIPTION = ""
URL = ""
EMAIL = ""
AUTHOR = ""
REQUIRES_PYTHON = ">=3.9.0"


# The rest you shouldn't have to touch too much. :)
# Except, perhaps the License and Trove Classifiers!
# ------------------------------------------------

long_description = DESCRIPTION

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'ride_duration'
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
    install_requires=["click==8.1.3; python_version >= '3.7'", 'flask==2.1.2', "importlib-metadata==4.11.4; python_version < '3.10'", "itsdangerous==2.1.2; python_version >= '3.7'", "jinja2==3.1.2; python_version >= '3.7'", "joblib==1.1.0; python_version >= '3.6'", "markupsafe==2.1.1; python_version >= '3.7'", "numpy==1.22.4; python_version >= '3.8'", 'scikit-learn==1.0.2', "scipy==1.8.1; python_version < '3.11' and python_version >= '3.8'", "threadpoolctl==3.1.0; python_version >= '3.6'", "werkzeug==2.1.2; python_version >= '3.7'", "zipp==3.8.0; python_version >= '3.7'"],
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)