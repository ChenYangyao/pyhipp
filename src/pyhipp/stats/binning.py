from __future__ import annotations
from typing import Any, Tuple
import numpy as np
from numba import njit
from pyhipp.core import DataDict, abc
from functools import cached_property

class Bin:
    @staticmethod
    def parse_p_range_spec(
        x_to_bin: np.ndarray = None, 
        range: Tuple[float,float] = None, 
        p_range: Tuple[float,float] = None) -> Tuple[float,float]:
        
        if x_to_bin is None:
            assert range is not None
            assert p_range is None
            return range
        
        x = np.asarray(x_to_bin)
        if range is None:
            if p_range is None:
                x_min, x_max = x.min(), x.max()
            else:
                x_min, x_max = np.quantile(x, p_range)
        else:
            x_min, x_max = range
            if p_range is not None:
                x_min1, x_max1 = np.quantile(x, p_range)
                x_min = max(x_min, x_min1)
                x_max = min(x_max, x_max1)
                
        return x_min, x_max
    
    @staticmethod
    def parse_sub_bin_spec(bins: int = 10, 
                           sub_bins: int = 1,
                           range: Tuple[float,float] = None):
        '''
        @sub_bins: bin refinement factor, i.e., resulting n_bins is the original 
        multiplying with `sub_bins`.
        
        If bins is scalar, return (n_bins after refinement, range, sub_bins).
        Otherwise return (edges after refinement, None, sub_bins)
        '''
        assert sub_bins >= 1
        
        if np.isscalar(bins):
            bins = bins * sub_bins
            assert range is not None
            min, max = range
        else:
            # bins are actually bin edges
            bins = Bin.insert_sub_bins_into_edges(bins, sub_bins)
            assert range is None
            min, max = bins[0], bins[-1]
            
        return bins, sub_bins, (min, max)

    @staticmethod
    def insert_sub_bins_into_edges(edges, sub_bins=1):
        e = np.asarray(edges)
        if sub_bins == 1:
            return e
        
        e_new = np.linspace(e[:-1], e[1:], sub_bins, 
                            endpoint=False).T.ravel()
        e_end = (e[-1], )
        return np.concatenate( (e_new, e_end) )
    
    
    @staticmethod
    def edges_to_centers(e, stride=1):
        e = np.array(e, copy=False)
        lo, hi = e[:-stride], e[stride:]
        return .5 * ( lo + hi )
    
    @staticmethod
    def edges_to_widths(e, stride=1):
        e = np.array(e, copy=False)
        lo, hi = e[:-stride], e[stride:]
        return hi - lo
    
    @staticmethod
    def combine_adjacent_bins(x, n_comb):
        '''
        @n_comb: int (>=1).
        
        Return an array sized x.size - n_comb + 1, whose each element is summed 
        from n_comb adjacent elements in x.
        
        If x.ndim > 1, the combination is along the first dimension.
        '''
        assert n_comb >= 1
        
        x = np.asarray(x)
        assert x.ndim >= 1
        n_out = len(x) - n_comb +1
        assert n_out > 0
        
        x_out = x[:n_out].copy()
        for b in range(1, n_comb):
            x_out += x[b:b+n_out]
        
        return x_out
    
class Hist1D:
    @staticmethod
    def from_overlapped_bins(x: np.ndarray, bins: int, sub_bins: int, 
        range: Tuple[float,float], weights: np.ndarray =None) -> DataDict:
        
        _h, _e = np.histogram(x, bins=bins, range=range, weights=weights)
        
        x = Bin.edges_to_centers(_e, sub_bins)
        dx = Bin.edges_to_widths(_e, sub_bins)
        h = Bin.combine_adjacent_bins(_h, sub_bins)
        
        return DataDict({'x': x, 'dx': dx, 'h': h, 
                         'sub_e': _e, 'sub_h': h}) 

        
class Bins(abc.HasSimpleRepr, abc.IsImmutable):
    '''
    IsImmutable: fields should not be changed after initialization.
    '''
    def __init__(self, **kw) -> None:
        
        super().__init__(**kw)
        
    def locate(self, x: np.ndarray) -> np.ndarray:
        '''
        Should return an array of integers, each of which is the index into the 
        bins, with x_edges[index] <= x < x_edges[index + 1].
        If x < x_edges[0], returns -1.
        If x >= x_edges[-1], returns len(x_edges) - 1, i.e., n_bins.
        '''
        raise NotImplementedError()
    
    def locate_bound(self, x: np.ndarray, lower=True, upper=True) -> np.ndarray:
        '''
        Returned is bound to a valid bin index, ranging in [0, n_bins).
        '''
        ind = self.locate(x)
        lower = 0 if lower else None
        upper = self.n_bins - 1 if upper else None
        ind = np.clip(ind, lower, upper)
        return ind
    
    def locate_nearest(self, x: np.ndarray) -> np.ndarray:
        '''
        Return a valid edge id, ranging in [0, n_edges), that locates the
        nearest edge to x.
        '''
        ind_l= self.locate_bound(x)
        ind_r = ind_l + 1
        e = self.x_edges
        x_l, x_r = e[ind_l], e[ind_r]
        d_l, d_r = x - x_l, x_r - x
        return np.where(d_l < d_r, ind_l, ind_r)
    
    @property
    def n_bins(self) -> int:
        raise NotImplementedError()
    
    @property
    def x_edges(self) -> np.ndarray:
        raise NotImplementedError()
    
    @cached_property
    def x_centers(self) -> np.ndarray:
        es = self.x_edges
        return 0.5 * (es[:-1] + es[1:])
    
class EqualSpaceBins(Bins):
    def __init__(self, x_range: Tuple[float,float] = (0., 1.), 
                 n_bins: int = 10, **kw) -> None:
        super().__init__(**kw)
        
        b, e = x_range
        assert b < e
        assert n_bins > 0
        x_span = e - b
        x_step = x_span / n_bins
        
        self._x_range = x_range
        self._n_bins = n_bins
        self._x_span = x_span
        self._x_step = x_step
        
    def locate(self, x: np.ndarray) -> np.ndarray:
        b, x_step, n_bins = self._x_range[0], self._x_step, self._n_bins
        ids = np.floor((x - b) / x_step).astype(np.int64)
        np.clip(ids, -1, n_bins, out=ids)
        return ids
        
    def to_simple_repr(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'x_range': self._x_range,
            'n_bins': self._n_bins,
            'x_span': self._x_span,
            'x_step': self._x_step,
        }
        
    @property
    def n_bins(self) -> int:
        return self._n_bins
    
    @cached_property
    def x_edges(self) -> np.ndarray:
        return np.linspace(*self._x_range, self._n_bins + 1)
    
    @property
    def x_range(self) -> Tuple[float, float]:
        return self._x_range
    
    @property
    def x_span(self) -> float:
        return self._x_span
    
    @property
    def x_step(self) -> float:
        return self._x_step
        
class BiSearchBins(Bins):
    def __init__(self, x_edges: np.ndarray, **kw) -> None:
        
        super().__init__(**kw)
        
        x_edges = np.sort(x_edges)
        assert x_edges.ndim == 1 and x_edges.size >= 2
        n_bins = len(x_edges) - 1
        x_span = x_edges[-1] - x_edges[0]
        x_step_mean = (x_edges[1:] - x_edges[:-1]).mean()
        
        
        self._x_edges = x_edges
        self._n_bins = n_bins
        self._x_span = x_span
        self._x_step_mean = x_step_mean
        
    def locate(self, x: np.ndarray) -> np.ndarray:
        x_edges = self._x_edges
        return np.searchsorted(x_edges, x, side='right') - 1
        
    def to_simple_repr(self) -> Any:
        return {
            'type': self.__class__.__name__,
            'x_edges': self._x_edges,
            'n_bins': self._n_bins,
            'x_span': self._x_span,
            'x_step_mean': self._x_step_mean,
        }
        
    @property
    def n_bins(self) -> int:
        return self._n_bins
        
    @property
    def x_edges(self) -> np.ndarray:
        return self._x_edges
            
class BinnedData(abc.HasSimpleRepr):
    def __init__(self, bins: Bins = None, dtype=float, 
                 shape=(), **kw) -> None:
        
        super().__init__(**kw)
        
        if bins is None:
            bins = EqualSpaceBins()
        
        d_shape = (bins.n_bins,) + shape
        
        self.bins = bins
        self.data = np.zeros(d_shape, dtype=dtype)
        
    def to_simple_repr(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'bins': self.bins.to_simple_repr(),
            'data': self.data,
        }
        
    def add(self, x: np.ndarray, weight=None):
        
        ids, data = self.bins.locate(x), self.data
        if weight is None:
            self.__add_unweighted(ids, data)
        else:
            self.__add_weighted(ids, weight, data)
            
    @staticmethod
    @njit
    def __add_unweighted(ids: np.ndarray, data: np.ndarray):
        n_bins = len(data)
        for id in ids:
            if id >= 0 and id < n_bins:
                data[id] += 1
                
    @staticmethod
    @njit
    def __add_weighted(ids: np.ndarray, weights: np.ndarray, data: np.ndarray):
        n_ids, n_bins = len(ids), len(data)
        assert n_ids == len(weights)
        for i in range(n_ids):
            id = ids[i]
            if id >= 0 and id < n_bins:
                data[id] += weights[i]
                
                
