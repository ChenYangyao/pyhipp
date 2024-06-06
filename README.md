# PyHIPP - Modern Python toolkit for HPC

[![Last commit](https://img.shields.io/github/last-commit/ChenYangyao/pyhipp/master)](https://github.com/ChenYangyao/pyhipp/commits/master)
[![Workflow Status](https://img.shields.io/github/actions/workflow/status/ChenYangyao/pyhipp/python-package.yml)](https://github.com/ChenYangyao/pyhipp/actions/workflows/python-package.yml)
[![MIT License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/ChenYangyao/pyhipp/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyhipp)](https://pypi.org/project/pyhipp/)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

PyHIPP is a computational toolkit, designed mainly for astronomical researches.
We pre-release this package, as our other projects depend extensively on it.

The package is still under active update, subjecting to structural changes and
functional extensions. The docs are still empty, and testings do not cover the 
full source tree, which we will continue to complete.


To install `pyhipp` and automatically handle the dependencies, use:
```bash
pip install pyhipp
```

## Usage 

See the Jupyter notebooks under `docs/`.
- `io.ipynb`: I/O facilities for data I/O, e.g., with HDF5.



## Interesting Features


**Schedulers for Parallel Computation**. A MPI-based job pool can be used like:
```py
from pyhipp.mpi import Comm, Pool

pool = Pool(Comm.world())
    
if pool.is_leader:
    for i in range(100):         # distribute works
        pool.assign_work(i)
    pool.join()
    print(pool.pop_replies())    # see gathered results
else:
    for work in pool.works():       
        i = work.content         # receive assigned data
        work.reply = np.sin(i)   # make response
```

**Cosmological Computation**. For example, the comoving distance at given redshifts
in a given cosmology:
```py
from pyhipp.astro.cosmology import model

cosm = model.predefined['tng']
z = [0., 1., 2., 3., 4.]
d = cosm.distances.comoving_at(z)   # => [0., 2300.371, 3597.988, ...]
```

**More is coming ...**