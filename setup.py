import setuptools
import pkg_resources
import pathlib
import os


req_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "requirements.txt")
with open(req_path) as f:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(f)
    ]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='democlassi',
    version='0.5.0',
    author='A. Alka M. Salissou',
    author_email='alkasalissou@hotmail.com',
    packages=setuptools.find_packages(exclude=['legacy']),
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*prototxt.txt', '*.caffemodel', '.html'],
    },
    url='https://github.com/AlkaSaliss/DEmoClassi',
    license='LICENSE',
    description='Collection of my python functions for training pytorch models to classify emotion, age, race, gender',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    classifiers=[
            "Programming Language :: Python :: 3.6",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
)
