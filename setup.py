from setuptools import setup

setup(
    name="lib_ml_remla24_team02",
    version="0.1.0",
    description="A package for pre-processing ML data for the REMLA course at the TU Delft.",
    author="REMLA 2024 Team 02",
    author_email="example@example.com",
    packages=["lib_ml_remla24_team02"],
    install_requires=[
        "scikit-learn",
        "tensorflow",
        "joblib"
    ]
)