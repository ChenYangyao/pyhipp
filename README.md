# PyHipp - Modern Python toolkit for HPC

[![Last commit](https://img.shields.io/github/last-commit/ChenYangyao/pyhipp/master)](https://github.com/ChenYangyao/pyhipp/commits/master)
[![Workflow Status](https://img.shields.io/github/actions/workflow/status/ChenYangyao/pyhipp/run-test.yml)](https://github.com/ChenYangyao/pyhipp/actions/workflows/run-test.yml)
[![MIT License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/ChenYangyao/pyhipp/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyhipp)](https://pypi.org/project/pyhipp/)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

PyHipp is a computational toolkit, designed mainly for astronomical researches.
We pre-release this package, as our other projects depend extensively on it.
The package is still under active update, subjecting to structural changes and
functional extensions. The docs are still incomplete, and testings do not cover the 
full source tree, which we will continue to complete.

To install `pyhipp` and automatically handle the dependencies, use:
```bash
$ pip install pyhipp
```

**Dependencies**

`pyhipp` depends on an GSL (GNU Scientific Library) installation. If you do not
have it installed, you can install it via conda:
```bash
$ conda install gsl
```
Alternatively, in Linux distributions such as Ubuntu, you can install it via the package manager:
```bash
$ sudo apt-get install libgsl-dev
```

## Usage 

See the Jupyter notebooks under `docs/`.
- `io.ipynb`: I/O facilities for data I/O, e.g., with HDF5.
- `astro.ipynb`: Astronomy-related functionalities, e.g. cosmological computations.