from __future__ import annotations
from sklearn.neighbors import NearestNeighbors
from ..core import DataDict
import numpy as np
from typing import Any, Union, Tuple, Mapping, Self, Iterable
import itertools
from . import reduction

class _KnnRegression:
    
    SingleReduce = reduction.Reduce | str | tuple[str, Mapping]
    Reduce = SingleReduce | Iterable[SingleReduce]
    
    avail_reduce = {
        'mean':     np.mean,
        'median':   np.median,
        'std':      np.std,
        'count':    lambda x: len(x),
        'qs':       lambda x, ps, **kw: np.quantile(x, ps, **kw),
    }
    
    def __init__(self, k = 32, reduce: Reduce = 'mean',
                 n_jobs: int = None, nn_kw = {}) -> None:
        '''
        @reduce: a reduction operation or a Iterable of them.
        @n_jobs: int. Number of jobs for parallel computation in 
            NearestNeighbors.
        '''
        impl = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, **nn_kw)
        
        self.impl               = impl
        self.y: np.ndarray      = None
        self.weight: np.ndarray = None
        self.reduce             = reduce
        
    def fit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray = None) -> Self:
        '''
        @X: array, shape >= 1.
        '''
        self.impl.fit(self.__cvt_x_shape(x))
        self.y = y
        self.weight = weight
        
    def __call__(self, x: np.ndarray, max_dx: float = None) -> DataDict:
        y_tr, w_tr, y_preds = self.y, self.weight, []
        rs = tuple(self.__iter_reduce())
        ds, ids = self.impl.kneighbors(self.__cvt_x_shape(x))
        for d, id in zip(ds, ids):
            if max_dx is None:
                y, w = y_tr[id], (None if w_tr is None else w_tr[id])
            else:
                sel = d < max_dx
                y, w = y_tr[id][sel], (None if w_tr is None else w_tr[id][sel])
            y_preds.append(tuple(op(y, w) for _, op in rs))
        y_preds = [
            np.array([y_pred[i] for y_pred in y_preds]) 
            for i, (key, _) in enumerate(rs)
        ]
        out = DataDict({'x': x, 'y': y_preds})
        for y_pred, (key, _) in zip(y_preds, rs):
            out[f'y_{key}'] = y_pred
        return out
        
    @staticmethod
    def __cvt_x_shape(x: np.ndarray): 
        if x.ndim == 1:
            x = x[:, None]
        return x
    
    @staticmethod
    def __single_reduce(reduce) -> Tuple[str, reduction.Reduce]:
        defs = reduction.predefined
        if isinstance(reduce, reduction.Reduce):
            out = reduce.key, reduce
        elif isinstance(reduce, str):
            out = reduce, defs[reduce]()
        else:
            key, kw = reduce
            out = key, defs[key](**kw)
        return out
    
    def __iter_reduce(self):
        reduce = self.reduce
        
        if isinstance(reduce, (reduction.Reduce, str, tuple)):
            yield self.__single_reduce(reduce)
            return
        
        for r in reduce:
            yield self.__single_reduce(r)

class KernelRegression1D:
    @staticmethod
    def by_knn(x: np.ndarray, y: np.ndarray, x_pred: np.ndarray, k: int = 32,
            max_dx: float = None, reduce: _KnnRegression.Reduce = 'mean', 
            weight: np.ndarray = None,
            n_jobs: int = None, nn_kw = {}) -> DataDict:
        
        knn = _KnnRegression(k=k, reduce=reduce,n_jobs=n_jobs, nn_kw=nn_kw)
        knn.fit(x, y, weight=weight)
        return knn(x_pred, max_dx=max_dx)
    
class KernelRegressionND:
    @staticmethod
    def by_knn(x: np.ndarray, y: np.ndarray, x_pred: np.ndarray, k: int = 32,
            max_dx: float = None, reduce: _KnnRegression.Reduce = 'mean', 
            weight: np.ndarray = None,
            n_jobs: int = None, nn_kw = {}) -> DataDict:
        
        knn = _KnnRegression(k=k, reduce=reduce, n_jobs=n_jobs, nn_kw=nn_kw)
        knn.fit(x, y, weight=weight)
        return knn(x_pred, max_dx=max_dx)