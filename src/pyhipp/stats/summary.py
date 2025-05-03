from __future__ import annotations
from typing import Any
import numpy as np
from functools import cached_property
from .reduction import Quantile
from pyhipp.core.abc import HasDictRepr
from dataclasses import dataclass, asdict
from copy import deepcopy

class SummaryFloatArray1D(HasDictRepr):
    
    repr_attr_keys = ('size', 'mean', 'stddev', 'median', '_1sigma', 
                      '_2sigma', '_3sigma')
    
    def __init__(self, a: np.ndarray):
        self.a = a
        
    @property
    def size(self) -> int:
        return len(self.a)
    
    @cached_property
    def mean(self) -> float:
        return self.a.mean()
    
    @cached_property
    def stddev(self) -> float:
        return self.a.std()
    
    @cached_property
    def median(self) -> float:
        return np.median(self.a)
    
    @cached_property
    def _1sigma(self) -> float:
        return Quantile('1sigma')(self.a)
    
    @cached_property
    def _2sigma(self) -> float:
        return Quantile('2sigma')(self.a)
    
    @cached_property
    def _3sigma(self) -> float:
        return Quantile('3sigma')(self.a)
    

class Summary:

    p_sigma_1 = [0.16, 0.84]
    p_sigma_2 = [0.025, 0.975]
    p_sigma_3 = [0.005, 0.995]

    @dataclass
    class FullResult:
        mean: np.ndarray
        sd: np.ndarray
        median: np.ndarray
        min: np.ndarray
        max: np.ndarray
        sigma_1: np.ndarray
        sigma_2: np.ndarray
        sigma_3: np.ndarray

        def as_dict(self) -> dict[str, Any]:
            r'''
            Convert to dict with deep copy.
            '''
            return asdict(self)

        @classmethod
        def from_dict(cls, d: dict[str, np.ndarray]):
            r'''
            Convert from dict to dataclass with deep copy.
            '''
            _d = {
                k: deepcopy(v) for k, v in d.items()
            }
            return cls(**_d)
            

    @staticmethod
    def on(vals: np.ndarray, axis=0):
        r'''
        Full summary of the input array of floating point values, `vals`.
        
        @vals: array-like of shape (N, ...B). By default (axis=0), 
        the first dimension (sized N) is reduced, and other batch dimension 
        (shaped B) is kept.
        
        @axis: axis to reduce.
        
        Attrs of returned result when axis=0:
        - mean, sd, median: shaped B.
        - sigma_1, sigma_2, sigma_3: shaped (2, B), the lower and upper 1, 2 
        and 3 sigma quantiles.
        
        When B = (), mean, sd and median are scalars.
        '''
        vals = np.asarray(vals)
        n_vals = len(vals)
        assert n_vals > 0
        if n_vals == 1:
            mean: np.ndarray = vals.mean(axis=axis)
            sd = np.zeros_like(mean)
            median = mean.copy()
            min = mean.copy()
            max = mean.copy()
            sigma_1 = np.array([mean, mean])
            sigma_2 = np.array([mean, mean])
            sigma_3 = np.array([mean, mean])
        else:
            mean = np.mean(vals, axis=axis)
            sd = np.std(vals, axis=axis)
            median = np.median(vals, axis=axis)
            min = np.min(vals, axis=axis)
            max = np.max(vals, axis=axis)
            sigma_1 = np.quantile(vals, Summary.p_sigma_1, axis=axis)
            sigma_2 = np.quantile(vals, Summary.p_sigma_2, axis=axis)
            sigma_3 = np.quantile(vals, Summary.p_sigma_3, axis=axis)
        return Summary.FullResult(
            mean, sd, median, min, max, sigma_1, sigma_2, sigma_3)
