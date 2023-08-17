from pyhipp.core import dataproc as dp
import pytest
import numpy as np


@pytest.fixture
def py_scalar():
    return 3

@pytest.fixture
def np_arr():
    return np.arange(10, dtype=int)

@pytest.fixture
def np_arr_ndim0():
    return np.array(3)

def test_bound_py_scalar(py_scalar: int):
    x = dp.Num.bound(py_scalar, lo=5)
    assert x == 5
    assert x is not py_scalar
    
    x = dp.Num.bound(py_scalar, hi=1)
    assert x == 1
    assert x is not py_scalar
    
    x = dp.Num.bound(py_scalar, lo=1, hi=4)
    assert x == 3
    assert x is py_scalar
    
    
def test_bound_np_arr(np_arr: np.ndarray):
    x = dp.Num.bound(np_arr, lo=4)
    assert np.all(x == [4,4,4,4,4, 5,6,7,8,9])
    assert x is not np_arr
    
    x = dp.Num.bound(np_arr, lo=4, hi = 7)
    assert np.all(x == [4,4,4,4,4, 5,6,7,7, 7])
    assert x is not np_arr
    
    x1 = np_arr.copy()
    x = dp.Num.bound(x1, copy=False)
    assert x is x1
    
    x = dp.Num.bound(x1, lo=4, copy=False)
    assert np.all(x == [4,4,4,4,4, 5,6,7,8,9])
    assert x is x1
    
    x = dp.Num.bound(x1, hi=7, copy=False)
    assert np.all(x == [4,4,4,4,4, 5,6,7,7,7])
    assert x is x1
    
    
def test_bound_np_arr_ndim0(np_arr_ndim0: np.ndarray):
    a = np_arr_ndim0
    
    a1 = dp.Num.bound(a, lo=4)
    assert a1 == 4
    assert a1 is not a
    
    a1 = dp.Num.bound(a, hi=1, copy=False)
    assert a1 == 1
    assert a1 is a


