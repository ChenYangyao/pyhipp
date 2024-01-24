from __future__ import annotations
import typing
from typing import Any, Tuple, ClassVar, Mapping
from pyhipp.core import abc, DataDict
import numpy as np

class Kernal(abc.HasName, abc.HasDictRepr):
    
    key: ClassVar[str] = None
    
    def __init__(self, **base_kw) -> None:
        super().__init__(**base_kw)
        
    def __call__(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d)
        return self._impl(d)
    
    def _impl(self, d: np.ndarray) -> np.ndarray:
        pass
    
class Gaussian(Kernal):
    
    key = 'gaussian'
    repr_attr_keys = ('sigma', )
    
    def __init__(self, sigma = 1.0, **base_kw) -> None:
        
        super().__init__(**base_kw)
        
        self.sigma = float(sigma)
        
    def _impl(self, d: np.ndarray) -> np.ndarray:
        sigma = self.sigma
        d_sq = d * d
        sigma_sq = sigma*sigma
        return np.exp( -0.5 * d_sq / sigma_sq )
    
class Tophat(Kernal):
    
    key = 'tophat'
    repr_attr_keys = ('radius', )
    
    def __init__(self, radius = 1.0, **base_kw) -> None:
        super().__init__(**base_kw)
        
        self.radius = float(radius)
        
    def _impl(self, d: np.ndarray) -> np.ndarray:
        out = d < self.radius
        out = np.array(out, dtype=float)
        return out
    
    def volume(self, n_dims: int, max_radius: float) -> float:
        if max_radius is not None:
            max_radius = min(max_radius, self.radius)
        return self._ball_volume(n_dims, max_radius)
    
    @staticmethod
    def _ball_volume(n_dims: int, radius: float) -> float:
        assert n_dims >= 0
        if n_dims == 0:
            vol = 1.0
        elif n_dims == 1:
            vol = 2.0 * radius
        else:
            vol_next = Tophat._ball_volume(n_dims-2, radius)
            vol = 2 * np.pi / n_dims * radius * radius * vol_next
        return vol
    
    
KernelSpec = Kernal | str | float | tuple[str, Mapping]
    
class _Predefined(DataDict):
    
    def __init__(self, **base_kw) -> None:
        
        super().__init__(**base_kw)
        
        kernels: list[Kernal] = [Gaussian, Tophat]
        self |= {
            k.key: k for k in kernels
        }
        
    def create(self, kernel: KernelSpec) -> tuple[str, Kernal]:
        if isinstance(kernel, Kernal):
            return kernel.key, kernel
        
        if isinstance(kernel, str):
            name, kw = kernel, {}
        elif np.isscalar(kernel):
            name, kw = 'gaussian', {'sigma': kernel}
        else:
            name, kw = kernel
        
        return name, self[name](**kw) 

        
predefined = _Predefined()