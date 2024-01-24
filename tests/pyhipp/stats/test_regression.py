from pyhipp.core import DataDict
from pyhipp.stats import KernelRegressionND, KernelRegression1D, random
import pytest
import numpy as np

@pytest.fixture
def regr_data_1d() -> DataDict:
    x = random.Rng().uniform(size=1000)
    x_pred = np.linspace(-2, 2, 16)
    y = np.sin(x)
    return DataDict({'x': x, 'y': y, 'x_pred': x_pred})


def test_knn(regr_data_1d: DataDict):
    x, y, x_pred = regr_data_1d['x', 'y', 'x_pred']
    out = KernelRegression1D.by_knn(x, y, x_pred=x_pred)
    print(out)
    
    out = KernelRegression1D.by_knn(x, y, x_pred=x_pred, 
        reduce=['mean', 'std', 'median'])
    print(out)