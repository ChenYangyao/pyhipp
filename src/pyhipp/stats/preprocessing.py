from __future__ import annotations
import typing
from typing import Tuple, ClassVar
from ..core import abc
import numpy as np

class Preprocess(abc.HasName, abc.HasDictRepr):
    
    key: ClassVar[str] = None
    
    def __init__(self, **base_kw) -> None:
        super().__init__(**base_kw)
        
    def __call__(self, x: np.ndarray, backward=False) -> np.ndarray:
        x = np.asarray(x)
        return  self._impl_backward(x) if backward else self._impl_forward(x)
    
    def _impl_forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    def _impl_backward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    
class Standardize(Preprocess):
    
    key = 'standardize'
    repr_attr_keys = ('mean', 'sd', 'sd_lb')
    
    def __init__(self, x_train: np.ndarray, sd_lb = 1.0e-10, **base_kw) -> None:
        super().__init__(**base_kw)
        
        x_train = np.asarray(x_train)
        assert x_train.ndim == 2
        
        mean: np.ndarray = x_train.mean(0)
        sd: np.ndarray = x_train.std(0)
        sd.clip(sd_lb, out=sd)
        
        self.mean = mean
        self.sd = sd
        
    def _impl_forward(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.sd
        
    def _impl_backward(self, x: np.ndarray) -> np.ndarray:
        return x * self.sd + self.mean