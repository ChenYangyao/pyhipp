from __future__ import annotations
from pyhipp.core import abc
from numba.experimental import jitclass
import numpy as np


@jitclass
class _Mesh:

    n_grids: int
    l_box:  float
    l_grid: float

    def __init__(self, n_grids=32, l_box=1.0) -> None:

        l_grid = l_box / n_grids

        self.n_grids = n_grids
        self.l_box = l_box
        self.l_grid = l_grid

    @property
    def total_n_grids(self) -> int:
        return self.n_grids**3

    @property
    def volume(self) -> float:
        return self.l_box**3

    @property
    def cell_volume(self) -> float:
        return self.l_grid**3
    
    def x_to_xi(self, x: float) -> int:
        l, n = self.l_grid, self.n_grids
        xi = np.int64(np.floor(x / l + 0.5)) % n
        return xi
        
    def x_to_xi_nd(self, x: np.ndarray) -> np.ndarray:
        '''
        x: arbitrary shaped.
        '''
        l, n = self.l_grid, self.n_grids
        xi = np.floor(x / l + 0.5).astype(np.int64) % n
        return xi

class Mesh(abc.HasDictRepr, abc.IsImmutable):

    repr_attr_keys = ('n_grids', 'l_box', 'l_grid')

    def __init__(self, impl: _Mesh) -> None:
        '''
        Should not be modifiled after creation.
        '''
        self._impl = impl

    @classmethod
    def new(cls, n_grids=32, l_box=1.0):
        impl = _Mesh(n_grids, l_box)
        return cls(impl)

    @property
    def n_grids(self) -> int:
        return self._impl.n_grids

    @property
    def l_box(self) -> float:
        return self._impl.l_box

    @property
    def l_grid(self) -> float:
        return self._impl.l_grid

    @property
    def total_n_grids(self) -> int:
        return self._impl.total_n_grids

    @property
    def volume(self) -> float:
        return self._impl.volume

    @property
    def cell_volume(self) -> float:
        return self._impl.cell_volume
