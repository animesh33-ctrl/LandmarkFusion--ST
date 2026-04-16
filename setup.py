from setuptools import setup, find_packages

setup(
    name="landmark_fusion_st",
    version="1.0.0",
    description="LandmarkFusion-ST: Multi-Stream Spatial-Temporal Sign Language Recognition",
    author="Animesh Palui, Kaushik Dutta",
    packages=find_packages(exclude=["scripts", "cache", "checkpoints"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "mediapipe>=0.10.9",
        "opencv-python>=4.9.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "tqdm>=4.66.0",
        "Pillow>=10.2.0",
        "transformers>=4.38.0",
    ],
)
