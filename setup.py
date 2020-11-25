from setuptools import setup, setuptools
import os

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="pretty_cm",
    version="0.0.1",
    author='Philip Huang',
    author_email="p208p2002@gmail.com",
    description="Confusion Matrix in Python: plot a pretty confusion matrix (like Matlab) in python using seaborn and matplotlib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/p208p2002/pretty-print-confusion-matrix",
    packages=setuptools.find_packages(),
    install_requires=[
       'pandas',
       'matplotlib',
       'seaborn'
    ],
    python_requires='>=3.5',
)