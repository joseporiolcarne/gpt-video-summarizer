from setuptools import setup, find_packages

setup(
    name="gpt_video_summarizer",
    version="0.1.0",
    description="A tool for transforming lengthy video or audio content into concise, digestible summaries using AI models.",
    author="Josep Oriol Carné",
    author_email="joseporiolcarne@proton.me",
    url="https://github.com/joseporiolcarne/gpt-video-summarizer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai",
        "moviepy",
        "python-dotenv",
        "pytubefix",
        "pyperclip",
        "validators",
    ],
    entry_points={
        "console_scripts": [
            "gpt-video-summarizer=src.main:main",  # Correctly points to src.main
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
