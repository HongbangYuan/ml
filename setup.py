
"""
setup the whole project.
"""

from setuptools import find_packages, setup

setup(
    name="ml",
    version="0.1",
    author="Yiqun Chen",
    url="https://github.com/YiqunChen1999/ml",
    description="Machine Learning Course Homework",
    # install_requires=[
    #     "yacs>=0.1.6",
    #     "pyyaml>=5.1",
    #     "av",
    #     "matplotlib",
    #     "termcolor>=1.1",
    #     "simplejson",
    # ],
    # packages=find_packages()
    packages=find_packages(exclude=("configs", "logs", "results")),
)