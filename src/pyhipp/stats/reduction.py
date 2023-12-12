from __future__ import annotations
import typing
from typing import Any, Tuple, ClassVar
from pyhipp.core import abc, DataDict
import numpy as np

class Reduce(abc.HasName, abc.HasDictRepr):
    
    key: ClassVar[str] = None
    
    def __init__(self, **base_kw) -> None:
        super().__init__(**base_kw)

    def __call__(self, x: np.ndarray, weight: np.ndarray = None) -> np.ndarray:
        x = np.asarray(x)
        if weight is None:
            out = self._impl_without_weight(x)
        else:
            weight = np.asarray(weight)
            weight = weight / weight.sum()
            out = self._impl_with_weight(x, weight)
        return out

    def _impl_without_weight(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def _impl_with_weight(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
class Count(Reduce):
    key = 'count'
    repr_attr_keys = ('normalize',)
    def __init__(self, normalize = True, **base_kw) -> None:
        super().__init__(**base_kw)
        
        self.normalize = bool(normalize)
        
    def __call__(self, x: np.ndarray, weight: np.ndarray = None) -> np.ndarray:
        x = np.asarray(x)
        if weight is None:
            return x.size
        
        if self.normalize:
            return x.size
        
        return np.sum(weight)
    
class Sum(Reduce):
    key = 'sum'
    repr_attr_keys = ('normalize',)
    def __init__(self, normalize = False, **base_kw) -> None:
        super().__init__(**base_kw)
        
        self.normalize = bool(normalize)
        
    def __call__(self, x: np.ndarray, weight: np.ndarray = None) -> np.ndarray:
        x = np.asarray(x)
        if weight is None:
            return x.sum()
        
        weight = np.asarray(weight)
        if self.normalize:
            weight = weight / weight.sum()
        
        return (x * weight).sum()
    
class Mean(Reduce):
    key = 'mean'
    def _impl_without_weight(self, x: np.ndarray) -> np.ndarray:
        return x.mean()
    
    def _impl_with_weight(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return (x * weight).sum()
    
class StdDev(Reduce):
    
    key = 'std_dev'
    
    def _impl_without_weight(self, x: np.ndarray) -> np.ndarray:
        return x.std()
    
    def _impl_with_weight(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        mean = Mean()
        x_mean = mean(x, weight)
        dx = x - x_mean
        x_sd = mean(dx * dx, weight)
        return x_sd

class Quantile(Reduce):
    key = 'quantile'
    repr_attr_keys = ('ps', 'sort')
    
    ps_map = {
        '1sigma': [0.16, 0.84],
        '2sigma': [0.025, 0.975],
        '3sigma': [0.005, 0.995],
        'median+1sigma': [0.16, 0.5, 0.84],
        'median+2sigma': [0.025, 0.5, 0.975],
        'median+3sigma': [0.005, 0.5, 0.995],
    }
    
    def __init__(self, ps: np.ndarray | str = 'median+1siamg', 
                 sort = True, **base_kw) -> None:
        super().__init__(**base_kw)
        if isinstance(ps, str):
            ps = self.ps_map[ps]
        self.ps = np.array(ps)
        self.sort = bool(sort)
        
    def _impl_without_weight(self, x: np.ndarray) -> np.ndarray:
        return np.quantile(x, self.ps)
    
    def _impl_with_weight(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        if self.sort:
            idx = np.argsort(x)
            x, weight = x[idx], weight[idx]
        cdf = np.cumsum(weight)
        qs = np.interp(self.ps, cdf, x)
        return qs

class Errorbar(Reduce):
    '''
    @reduce: 'mean+sd', 'median+1sigma', 'median+2sigma', 'median+3sigma'.
    '''
    key = 'errorbar'
    repr_attr_keys = ('reduce', )
    
    def __init__(self, reduce='median+1sigma', **base_kw) -> None:
        super().__init__(**base_kw)

        if reduce == 'mean+sd':
            op = self.__op_mean_sd()
        else:
            assert reduce in Quantile.ps_map
            op = lambda x, weight: self.__op_median_sigma(reduce, x, weight)

        self.reduce = reduce
        self.op = op
        
    def __call__(self, x: np.ndarray, weight: np.ndarray = None) -> np.ndarray:
        return self.op(x, weight)

    def __op_mean_sd(self, x: np.ndarray, weight: np.ndarray):
        mean, sd = Mean()(x, weight), StdDev()(x, weight)
        return np.array([mean, sd, sd])
    
    def __op_median_sigma(self, ps: str, x: np.ndarray, weight: np.ndarray):
        x_lo, x_med, x_hi = Quantile(ps)(x, weight)
        return x_med, x_med - x_lo, x_hi - x_med

class Median(Reduce):
    
    key = 'median'
    
    repr_attr_keys = ('sort',)
    
    def __init__(self, sort = True, **base_kw) -> None:
        super().__init__(**base_kw)
        self.sort = bool(sort)
        
    def _impl_without_weight(self, x: np.ndarray) -> np.ndarray:
        return np.median(x)
    
    def _impl_with_weight(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        quantile = Quantile(ps = 0.5, sort=self.sort)
        return quantile(x, weight)

class _Predefined(DataDict):
    
    def __init__(self, **base_kw) -> None:
        
        super().__init__(**base_kw)
        
        self |= {
            'count': Count,
            'cnt': Count,
            'sum': Sum,
            'mean': Mean,
            'std_dev': StdDev,
            'std': StdDev,
            'sd': StdDev,
            'quantile': Quantile,
            'qs': Quantile,
            'median': Median,
            'errorbar': Errorbar,
        }
    
predefined = _Predefined()

