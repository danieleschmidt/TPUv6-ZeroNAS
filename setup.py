from setuptools import setup, find_packages

setup(
    name="tpuv6-zeronas",
    version="0.1.0",
    description="Neural Architecture Search optimization for TPUv6 hardware",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.com",
    packages=find_packages(),
    install_requires=[
        # Core functionality works with Python standard library only
    ],
    extras_require={
        "full": [
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "matplotlib>=3.5.0",
            "tqdm>=4.62.0",
        ],
        "ml": [
            "torch>=1.12.0",
            "tensorflow>=2.8.0",
            "jax>=0.3.0",
            "jaxlib>=0.3.0",
        ],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)