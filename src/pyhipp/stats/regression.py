from __future__ import annotations
from sklearn.neighbors import NearestNeighbors
from ..core import DataDict
import numpy as np
from typing import Union, Tuple

class KernelRegression1D:
    
    avail_reduce = {
        'mean': np.mean,
        'median': np.median,
        'count': lambda x: len(x),
        'qs': lambda x, ps: np.quantile(x, ps),
    }
    
    @staticmethod
    def by_knn(x: np.ndarray, y: np.ndarray, x_pred: np.ndarray, n: int = 32,
            max_dx: float = None,
            reduce: Union[str, Tuple[str,...]] = 'mean', reduce_kw = {},
            n_jobs: int = None) -> DataDict:
        
        if isinstance(reduce, str):
            reduce = (reduce,)
            reduce_kw = (reduce_kw,)
        avail = KernelRegression1D.avail_reduce
        reduce_fn = tuple(avail[r] for r in reduce)
        
        knn = NearestNeighbors(n_neighbors=n, n_jobs=n_jobs)
        knn.fit(x[:, None])
        
        y_pred = []
        diss, ids = knn.kneighbors(x_pred[:, None])
        for dis, id in zip(diss, ids):
            if max_dx is not None:
                sel = dis < max_dx
                uid = id[sel]
            else:
                uid = id
            uy = y[uid]
            y_pred.append(
                tuple(fn(uy, **kw) for fn, kw in zip(reduce_fn, reduce_kw))  
            ) 
        
        out = DataDict({
            'x': x_pred,
        })
        for i, r in enumerate(reduce):
            out[f'y_{r}'] = np.array([yp[i] for yp in y_pred]) 
        return out