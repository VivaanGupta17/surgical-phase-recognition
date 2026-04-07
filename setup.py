"""SurgPhase package setup configuration."""

from pathlib import Path

from setuptools import find_packages, setup

# Read long description from README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Core runtime requirements (no dev/test extras)
install_requires = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "pandas>=2.0.0",
    "PyYAML>=6.0",
    "scikit-learn>=1.3.0",
    "einops>=0.7.0",
    "tqdm>=4.66.0",
    "onnx>=1.15.0",
]

extras_require = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.1.0",
        "mypy>=1.6.0",
        "pre-commit>=3.5.0",
    ],
    "notebooks": [
        "jupyter>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "ipywidgets>=8.1.0",
    ],
    "deploy": [
        "onnxruntime-gpu>=1.16.0",
        "onnx-simplifier>=0.4.33",
        "ultralytics>=8.0.0",
    ],
    "logging": [
        "wandb>=0.16.0",
        "tensorboard>=2.14.0",
    ],
}

# Convenience 'all' extra
extras_require["all"] = list(
    set(dep for extra in extras_require.values() for dep in extra)
)

setup(
    name="surgphase",
    version="0.1.0",
    author="SurgPhase Contributors",
    author_email="surgphase@example.com",
    description=(
        "Real-time deep learning for surgical phase recognition "
        "and instrument detection in laparoscopic cholecystectomy"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/surgical-phase-recognition",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/surgical-phase-recognition/issues",
        "Documentation": "https://github.com/yourusername/surgical-phase-recognition/docs",
    },
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords=[
        "surgical AI",
        "phase recognition",
        "instrument detection",
        "cholecystectomy",
        "laparoscopic surgery",
        "temporal convolutional network",
        "computer-assisted surgery",
    ],
    entry_points={
        "console_scripts": [
            "surgphase-train=scripts.train:main",
            "surgphase-evaluate=scripts.evaluate:main",
            "surgphase-demo=scripts.demo_realtime:main",
            "surgphase-export=scripts.export_onnx:main",
        ],
    },
    include_package_data=True,
    package_data={
        "surgphase": ["configs/*.yaml"],
    },
)
