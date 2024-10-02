from setuptools import setup, find_packages

setup(
    name="npcsh",
    version="0.1.1",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "jinja2",
        "pandas",
        "ollama",
        "requests",
        "PyYAML",
        "openai-whisper",
        "pyaudio",
    ],
    entry_points={
        "console_scripts": [
            "npcsh=npcsh.npcsh:main",
        ],
    },
    author="Christopher Agostino",
    author_email="cjp.agostino@example.com",
    description="A way to use npcsh",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cagostino/npcsh",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
)
