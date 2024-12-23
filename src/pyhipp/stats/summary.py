from __future__ import annotations
import numpy as np
from functools import cached_property
from .reduction import Quantile
from pyhipp.core.abc import HasDictRepr

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