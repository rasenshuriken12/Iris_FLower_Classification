from setuptools import setup, find_packages

setup(
    name="iris-classification",
    version="0.1.0",
    author="Your Name",
    description="Iris Flower Classification ML project",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
    ],
)