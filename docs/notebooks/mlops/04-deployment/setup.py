from pathlib import Path
from setuptools import find_packages, setup


# Package meta-data.
NAME = "ride-duration-prediction"
PACKAGE_NAME = "ride_duration"
DESCRIPTION = "Predicting ride duration for TLC Trip Record Data."
URL = "https://particle1331.github.io/inefficient-networks/notebooks/mlops/04-deployment/notes.html"
EMAIL = "particle1331@gmail.com"
AUTHOR = "Ron Medina"
REQUIRES_PYTHON = ">=3.9.0,<3.10"
INSTALL_REQUIRES = [
    "scikit-learn==1.0.2", 
    "mlflow>=1.26.1,<1.27.0", 
    "pandas>=1.4.3,<1.5.0",
    "joblib>=1.1.0,<1.2.0"
]

# The rest you shouldn't have to touch too much. :)
# Except, perhaps the License and Trove Classifiers!
# ------------------------------------------------

long_description = DESCRIPTION

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / PACKAGE_NAME
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
    package_data={PACKAGE_NAME: ["VERSION"]},
    install_requires=INSTALL_REQUIRES,
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
