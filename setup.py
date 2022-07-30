"""
References:
    - https://setuptools.pypa.io/en/latest/userguide/quickstart.html
    - https://packaging.python.org/tutorials/packaging-projects/
"""
from pathlib import Path
from pkg_resources import parse_requirements
from setuptools import setup, find_packages

NAME = "lomltk"
VERSION = "0.0.1"
DESCRIPTION = "The lord of Machine Learning toolkits"


def read(*names, **kwargs):
    kwargs.setdefault("encoding", "utf8")

    with Path(*names).open(**kwargs) as f:
        return f.read()


def read_requirements(*names, **kwargs):
    kwargs.setdefault("encoding", "utf8")

    with Path(*names).open(**kwargs) as f:
        return [str(req) for req in parse_requirements(f)]


# Setting up
setup(
    name=NAME,
    version=VERSION,
    author="Zijian Hu",
    author_email="zijian-hu@outlook.com",
    description=DESCRIPTION,
    url="https://github.com/zijian-hu/lomltk",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["lomltk"]),
    install_requires=read_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
