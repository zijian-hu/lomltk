"""
References:
    - https://setuptools.pypa.io/en/latest/userguide/quickstart.html
    - https://packaging.python.org/tutorials/packaging-projects/
"""
from pathlib import Path
from pkg_resources import parse_requirements
import platform
import re
from setuptools import setup, find_packages
import sys

NAME = "lomltk"
DESCRIPTION = "The lord of Machine Learning toolkits"
MIN_PYTHON_VERSION = (3, 7, 0)

if sys.version_info < MIN_PYTHON_VERSION:
    print(f"You are using Python {platform.python_version()}. Python >= {MIN_PYTHON_VERSION} is required.")
    sys.exit(-1)


def read(*paths, **kwargs):
    kwargs.setdefault("encoding", "utf8")

    with Path(*paths).open(**kwargs) as f:
        return f.read()


def read_requirements(*paths, **kwargs):
    kwargs.setdefault("encoding", "utf8")

    with Path(*paths).open(**kwargs) as f:
        return [str(req) for req in parse_requirements(f)]


def read_version(*paths, **kwargs):
    version_file = read(*paths, **kwargs)
    match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", version_file)

    if match is not None:
        return match.group(1)
    else:
        return "UNKNOWN"


# Setting up
setup(
    name=NAME,
    version=read_version("lomltk/__init__.py"),
    author="Zijian Hu",
    author_email="zijian-hu@outlook.com",
    description=DESCRIPTION,
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/zijian-hu/lomltk",
    packages=find_packages(include=["lomltk"]),
    include_package_data=True,
    install_requires=read_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7"
)
