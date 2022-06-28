from ineff.config import config

VERSION_PATH = config.PACKAGE_ROOT / "VERSION"
name = "regression_model"

with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()
