from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as handle:
    long_description = handle.read()

setup(
    name="otfs-ris-simulator",
    version="0.1.0",
    description="OTFS simulator with RIS optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={"dev": ["pytest>=6.2.0", "pytest-cov>=2.12.0"]},
)
