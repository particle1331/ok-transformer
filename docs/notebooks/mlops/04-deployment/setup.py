from pathlib import Path
from setuptools import find_packages, setup


# Package meta-data.
NAME = "ride-duration-prediction"
DESCRIPTION = "Predicting ride duration for TLC Trip Record Data."
URL = "https://particle1331.github.io/inefficient-networks/notebooks/mlops/04-deployment/notes.html"
EMAIL = "particle1331@gmail.com"
AUTHOR = "Ron Medina"
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
    install_requires=["alembic==1.8.0; python_version >= '3.7'", 'boto3==1.24.17', "botocore==1.27.17; python_version >= '3.7'", "certifi==2022.6.15; python_version >= '3.6'", "charset-normalizer==2.0.12; python_full_version >= '3.5.0'", "click==8.1.3; python_version >= '3.7'", "cloudpickle==2.1.0; python_version >= '3.6'", 'databricks-cli==0.17.0', "docker==5.0.3; python_version >= '3.6'", "entrypoints==0.4; python_version >= '3.6'", 'flask==2.1.2', "gitdb==4.0.9; python_version >= '3.6'", "gitpython==3.1.27; python_version >= '3.7'", "gunicorn==20.1.0; platform_system != 'Windows'", "idna==3.3; python_full_version >= '3.5.0'", "importlib-metadata==4.12.0; python_version < '3.10'", "itsdangerous==2.1.2; python_version >= '3.7'", "jinja2==3.1.2; python_version >= '3.7'", "jmespath==1.0.1; python_version >= '3.7'", "joblib==1.1.0; python_version >= '3.6'", "mako==1.2.0; python_version >= '3.7'", "markupsafe==2.1.1; python_version >= '3.7'", 'mlflow==1.26.1', "numpy==1.23.0; python_version >= '3.8'", "oauthlib==3.2.0; python_version >= '3.6'", "packaging==21.3; python_version >= '3.6'", 'pandas==1.4.3', "prometheus-client==0.14.1; python_version >= '3.6'", 'prometheus-flask-exporter==0.20.2', "protobuf==4.21.2; python_version >= '3.7'", "pyjwt==2.4.0; python_version >= '3.6'", "pyparsing==3.0.9; python_full_version >= '3.6.8'", "python-dateutil==2.8.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", 'pytz==2022.1', "pyyaml==6.0; python_version >= '3.6'", 'querystring-parser==1.2.4', "requests==2.28.0; python_version >= '3.7' and python_version < '4'", "s3transfer==0.6.0; python_version >= '3.7'", 'scikit-learn==1.0.2', "scipy==1.8.1; python_version < '3.11' and python_version >= '3.8'", "setuptools==62.6.0; python_version >= '3.7'", "six==1.16.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", "smmap==5.0.0; python_version >= '3.6'", "sqlalchemy==1.4.39; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5'", "sqlparse==0.4.2; python_full_version >= '3.5.0'", "tabulate==0.8.10; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'", "threadpoolctl==3.1.0; python_version >= '3.6'", "urllib3==1.26.9; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4' and python_version < '4'", "websocket-client==1.3.3; python_version >= '3.7'", "werkzeug==2.1.2; python_version >= '3.7'", "zipp==3.8.0; python_version >= '3.7'"],
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