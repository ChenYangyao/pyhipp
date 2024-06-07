from pyhipp.core import DataDict
from pyhipp.stats import KernelRegressionND, KernelRegression1D, Rng
import pytest
import numpy as np

@pytest.fixture
def regr_data_1d() -> DataDict:
    x = Rng().uniform(size=1000)
    x_pred = np.linspace(-2, 2, 16)
    y = np.sin(x)
    return DataDict({'x': x, 'y': y, 'x_pred': x_pred})

@pytest.fixture
def regr_data_2d() -> DataDict:
    rng = Rng()
    x = rng.uniform(size=(1000, 2))
    x_pred = rng.uniform(size=(16, 2))
    y = np.sin(x[:,0]) + np.cos(x[:,1])
    return DataDict({'x': x, 'y': y, 'x_pred': x_pred})
    
def test_knn(regr_data_1d: DataDict):
    x, y, x_pred = regr_data_1d['x', 'y', 'x_pred']
    out = KernelRegression1D.by_knn(x, y, x_pred=x_pred)
    print(out)
    
    out = KernelRegression1D.by_knn(x, y, x_pred=x_pred, 
        reduce=['mean', 'std', 'median'])
    print(out)
    
def test_knn_2d(regr_data_2d: DataDict):
    x, y, x_pred = regr_data_2d['x', 'y', 'x_pred']
    out = KernelRegressionND.by_knn(x, y, x_pred=x_pred)
    print(out)
    
    out = KernelRegressionND.by_knn(x, y, x_pred=x_pred, 
        reduce=['mean', 'std', 'median'])
    print(out)
    
def test_local_kernel(regr_data_1d: DataDict):
    x, y, x_pred = regr_data_1d['x', 'y', 'x_pred']
    out = KernelRegression1D.by_local_kernel(x, y, x_pred=x_pred)
    print(out)
    
    out = KernelRegression1D.by_local_kernel(x, y, x_pred=x_pred, 
        reduce=['mean', 'std', 'median'])
    print(out)
    
def test_local_kernel_2d(regr_data_2d: DataDict):
    x, y, x_pred = regr_data_2d['x', 'y', 'x_pred']
    out = KernelRegressionND.by_local_kernel(x, y, x_pred=x_pred)
    print(out)
    
    out = KernelRegressionND.by_local_kernel(x, y, x_pred=x_pred, 
        reduce=['mean', 'std', 'median'])
    print(out)