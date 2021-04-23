from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clustereval", # Replace with your own username
    version="0.0.2",
    author="Vinay Swamy",
    author_email="swamyvinny@gmail.com",
    description="Evaluating accuracy of graph based clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinay-swamy/clustereval",
    project_urls={
        "Bug Tracker": "https://github.com/vinay-swamy/clustereval/issues",
    },
    python_requires=">=3.6",
    packages=['clustereval'],
    setup_requires=['numpy', 'pybind11', 'setuptools-rust'],
    install_requires=['pybind11', 'numpy', 'scipy', 'pandas', 'hnswlib', 'python-igraph', 'leidenalg>=0.7.0', 'louvain','umap-learn'],
    rust_extensions=[RustExtension(
        "clustereval._calc_metrics", "Cargo.toml")],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False
)
