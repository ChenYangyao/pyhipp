from __future__ import annotations
from sklearn.neighbors import NearestNeighbors
from ..core import DataDict
import numpy as np
from typing import Any, Union, Tuple, Mapping, Self
import itertools

class _KnnRegression:
    
    avail_reduce = {
        'mean':     np.mean,
        'median':   np.median,
        'std':      np.std,
        'count':    lambda x: len(x),
        'qs':       lambda x, ps, **kw: np.quantile(x, ps, **kw),
    }
    
    def __init__(self, k = 32, reduce: Union[str, Tuple[str, ...]] = 'mean',
                 reduce_kw: Union[Mapping, Tuple[Mapping, ...]] = {},
                 n_jobs: int = None) -> None:
        
        impl = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)
        
        self.impl = impl
        self.y: np.ndarray = None
        self.reduce = reduce
        self.reduce_kw = reduce_kw
        
    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        '''
        @X: array, shape >= 1.
        '''
        self.impl.fit(self.__cvt_x_shape(x))
        self.y = y
        
    def __call__(self, x: np.ndarray, max_dx: float = None) -> DataDict:
        y_trs, y_preds = self.y, []
        reduces = tuple(self.__iter_reduce())
        ds, ids = self.impl.kneighbors(self.__cvt_x_shape(x))
        for d, id in zip(ds, ids):
            y = y_trs[id]
            if max_dx is not None:
                y = y[d<max_dx]
            y_preds.append(tuple(
                fn(y, **kw) for _, kw, fn in reduces
            ))
        out = DataDict({'x': x,})
        for i, (r, *_) in enumerate(reduces):
            out[f'y_{r}'] = np.array([y[i] for y in y_preds]) 
        return out

        
    @staticmethod
    def __cvt_x_shape(x: np.ndarray): 
        if x.ndim == 1:
            x = x[:, None]
        return x
    
    def __iter_reduce(self):
        rs, kws = self.reduce, self.reduce_kw
        avail = self.avail_reduce
        if isinstance(rs, str):
            assert isinstance(kws, Mapping)
            fns = avail[rs]
            yield rs, kws, fns
            return    
        
        if isinstance(kws, Mapping):
            kws = itertools.cycle((kws,))
        for r, kw in zip(rs, kws):
            fn = avail[r]
            yield r, kw, fn

class KernelRegression1D:
    @staticmethod
    def by_knn(x: np.ndarray, y: np.ndarray, x_pred: np.ndarray, k: int = 32,
            max_dx: float = None,
            reduce: Union[str, Tuple[str,...]] = 'mean', reduce_kw = {},
            n_jobs: int = None) -> DataDict:
        
        knn = _KnnRegression(k=k, reduce=reduce, reduce_kw=reduce_kw, 
                             n_jobs=n_jobs)
        knn.fit(x, y)
        return knn(x_pred, max_dx=max_dx)
    
class KernelRegressionND:
    @staticmethod
    def by_knn(x: np.ndarray, y: np.ndarray, x_pred: np.ndarray, k: int = 32,
            max_dx: float = None,
            reduce: Union[str, Tuple[str,...]] = 'mean', reduce_kw = {},
            n_jobs: int = None) -> DataDict:
        
        knn = _KnnRegression(k=k, reduce=reduce, reduce_kw=reduce_kw, 
                             n_jobs=n_jobs)
        knn.fit(x, y)
        return knn(x_pred, max_dx=max_dx)