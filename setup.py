from setuptools import setup, find_packages

setup(
    name="biomorphx",
    version="1.0.0",
    description="Cross-Scale Mechano-Genomic, Biomaterial, Nanofabrication & Life-Critical Biomedical Engineering SDK",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="BioMorphX Research Consortium",
    license="MIT",
    url="https://github.com/Runtime-Slayers/BioMorphX",
    project_urls={
        "Source": "https://github.com/Runtime-Slayers/BioMorphX",
        "Documentation": "https://github.com/Runtime-Slayers/BioMorphX#readme",
        "Bug Reports": "https://github.com/Runtime-Slayers/BioMorphX/issues",
    },
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    entry_points={
        "console_scripts": [
            "biomorphx=biomorphx.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
