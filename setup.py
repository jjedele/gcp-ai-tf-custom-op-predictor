from setuptools import setup

setup(
    name="custom-op-tf-predictor",
    version="0.1",
    scripts=["predictor.py"],
    install_requires=[
        "tf-sentencepiece==0.1.82",
    ],
)
