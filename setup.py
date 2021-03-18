import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clustereval", # Replace with your own username
    version="0.0.1",
    author="Vinay Swamy",
    author_email="swamyvinny@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinay-swamy/clustereval",
    project_urls={
        "Bug Tracker": "https://github.com/vinay-swamy/clustereval/issues",
    },
    python_requires=">=3.6",
    setup_requires=['numpy', 'pybind11'],
    install_requires=['pybind11', 'numpy', 'scipy', 'pandas', 'hnswlib', 'python-igraph', 'leidenalg>=0.7.0', 'louvain'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
