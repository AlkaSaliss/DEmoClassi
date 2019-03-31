import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='democlassi',
    version='0.2.3',
    author='A. Alka M. Salissou',
    author_email='alkasalissou@hotmail.com',
    packages=setuptools.find_packages(exclude='legacy'),
    url='https://github.com/AlkaSaliss/DEmoClassi',
    license='LICENSE',
    description='Collection of my python functions for training pytorch models to classify emotion, age, race, gender',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-ignite",
        "imutils",
        "dlib",
        "kaggle",
        "tensorboardX",
    ],
    classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
)
