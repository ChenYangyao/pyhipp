from pyhipp.astro.cosmology import model
import numpy as np


def test_has_predefined():
    print(model.predefined)
    
    tng = model.predefined['tng']
    print(tng)
    
    eagle = model.predefined['eagle']
    print(eagle)
    
def test_simple_repr():
    m = model.predefined['tng']
    assert isinstance(m.to_simple_repr(), dict)

    ht = m.halo_theory
    assert isinstance(ht, model.HaloTheory)
    assert isinstance(ht.to_simple_repr(), dict)
    
    ldc = ht.lg_delta_c([3., 1., 0.])
    assert isinstance(ldc, np.ndarray)
    assert len(ldc) == 3 and ldc.ndim == 1
    
    ldc = ht.lg_delta_c(1.)
    assert isinstance(ldc, np.ndarray)
    assert ldc.ndim == 0
    
def test_distances():
    cosm = model.predefined['tng']
    z = [0., 1., 2., 3., 4.]
    d = cosm.distances.comoving_at(z)
    print(d)