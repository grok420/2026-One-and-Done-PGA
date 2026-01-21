"""Setup script for PGA One and Done Optimizer."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pga-one-and-done",
    version="1.0.0",
    author="Eric",
    author_email="gitberge@gmail.com",
    description="A comprehensive PGA fantasy golf One and Done optimization tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gitberge/2026-One-and-Done-PGA",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Games/Entertainment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "selenium>=4.0.0",
        "webdriver-manager>=4.0.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
    ],
    extras_require={
        "tts": ["pyttsx3>=2.90"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0"],
    },
    entry_points={
        "console_scripts": [
            "pga-oad=pga_one_and_done.cli:main",
        ],
    },
)
