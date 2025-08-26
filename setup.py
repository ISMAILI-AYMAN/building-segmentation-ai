from setuptools import setup, find_packages

with open("README_GITHUB.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="building-segmentation-ai",
    version="1.0.0",
    author="Building Segmentation AI Team",
    author_email="contact@buildingsegmentation.ai",
    description="A comprehensive deep learning solution for automated building segmentation from aerial imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/building-segmentation-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "building-segmentation=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.conf"],
    },
    keywords="deep-learning computer-vision segmentation building-detection aerial-imagery pytorch fastapi flask docker",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/building-segmentation-ai/issues",
        "Source": "https://github.com/yourusername/building-segmentation-ai",
        "Documentation": "https://github.com/yourusername/building-segmentation-ai/blob/main/README_GITHUB.md",
    },
)
