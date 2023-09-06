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

## Interesting Features

**Extended HDF5 IO**. Suppose you have a nested catalog data, e.g.,
```py
halo_cat = {
    'Header': {
        'n_subhalos': 10, 'n_halos': 5, 'version': '1.0.0',
        'source': 'ELUCID simulation', 'last_update': '2023-08-17',
    },
    'Subhalos': {
        'id': np.arange(10),
        'x': np.random.uniform(size=(10,3)), 
        'v': np.random.uniform(size=(10,3)),
    },
    'Halos': {
        'id': np.arange(5),
        'x': np.random.uniform(size=(5,3)),
        'v': np.random.uniform(size=(5,3)),
    },
}
```

Dump it recursively into a HDF5 file:
```py
from pyhipp.io import h5

h5.File.dump_to(path, halo_cat)
```
    
Load back all or a subset:
```py
halo_cat = h5.File.load_from(path)
halos = h5.File.load_from(path, 'Halos')
```
    
Of course, you can open the file, and load datasets separately:
```py
with h5.File(path) as f:
    dsets = f['Halos'].datasets
    x = dsets.x                   # load via attributes (Thanks Zhaozhou Li for the idea)
    id, v = dsets['id', 'v']      # load via getitem
    
    halos = f['Halos'].load()     # load all halos as a dict-like object
    x = halos['x']
```

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